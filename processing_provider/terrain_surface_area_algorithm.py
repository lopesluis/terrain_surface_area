# -*- coding: utf-8 -*-
import math
import numpy as np

from qgis.PyQt.QtCore import QCoreApplication, QVariant
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFeatureSink,
    QgsFields,
    QgsField,
    QgsGeometry,
    QgsPoint,            # <-- FIX: use QgsPoint for QGIS 3.34 LTR
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterLayer,
    QgsProcessingUtils,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsWkbTypes,
)


class TerrainSurfaceAreaAlgorithm(QgsProcessingAlgorithm):

    # ---- parameter names (stable API) ----
    INPUT_POLYGONS = 'INPUT_POLYGONS'
    INPUT_DEM = 'INPUT_DEM'
    SLOPE_METHOD = 'SLOPE_METHOD'
    MIN_DEM_COV_PCT = 'MIN_DEM_COV_PCT'
    VERT_SCALE = 'VERT_SCALE'
    OUTPUT = 'OUTPUT'

    # slope methods
    HORN = 0
    ZEVENBERGEN_THORNE = 1

    # output fields
    F_AREA_2D = 'AREA_2D_M2'
    F_AREA_SURF = 'AREA_SURF_M2'
    F_RATIO = 'SURF_RATIO'
    F_COV = 'DEM_COV_PCT'

    def tr(self, string):
        return QCoreApplication.translate('Processing', string)

    def name(self):
        return 'terrain_surface_area_dem'

    def displayName(self):
        return self.tr('Terrain Surface Area (DEM)')

    def group(self):
        return self.tr('Terrain analysis')

    def groupId(self):
        return 'terrain_analysis'

    def shortHelpString(self):
        return self.tr(
            "Computes planar area and terrain surface area of polygons using a DEM.\n"
            "(Calcula a área plana e a área real do terreno de polígonos usando um MDE)\n\n"
            "Outputs:\n"
            f"- {self.F_AREA_2D}: planar area (área plana)\n"
            f"- {self.F_AREA_SURF}: terrain surface area (área de superfície do terreno)\n"
            f"- {self.F_RATIO}: surface/planar ratio (relação superfície/plana)\n"
            f"- {self.F_COV}: valid DEM coverage (%) (cobertura válida do MDE)\n\n"
            "Slope method:\n"
            "Horn — recommended for most DEMs (Horn — recomendado para a maioria dos MDEs)\n"
            "Zevenbergen & Thorne — smoother terrain, may be sensitive to noise "
            "(Zevenbergen & Thorne — mais suave, pode ser sensível a ruído)\n"
        )

    def createInstance(self):
        return TerrainSurfaceAreaAlgorithm()

    # --------------------------- parameters ---------------------------

    def initAlgorithm(self, config=None):

        self.addParameter(
            QgsProcessingParameterFeatureSource(
                self.INPUT_POLYGONS,
                self.tr('Input polygons'),
                [QgsProcessing.TypeVectorPolygon],
            )
        )

        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT_DEM,
                self.tr('Elevation raster (DEM)'),
            )
        )

        self.addParameter(
            QgsProcessingParameterEnum(
                self.SLOPE_METHOD,
                self.tr('Slope method'),
                options=['Horn', 'Zevenbergen & Thorne'],
                defaultValue=self.HORN,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.MIN_DEM_COV_PCT,
                self.tr('Minimum DEM coverage (%)'),
                QgsProcessingParameterNumber.Double,
                defaultValue=70.0,
                minValue=0.0,
                maxValue=100.0,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                self.VERT_SCALE,
                self.tr('Vertical scale factor'),
                QgsProcessingParameterNumber.Double,
                defaultValue=1.0,
                minValue=0.000001,
            )
        )

        self.addParameter(
            QgsProcessingParameterFeatureSink(
                self.OUTPUT,
                self.tr('Output layer'),
            )
        )

    # --------------------------- core logic ---------------------------

    def processAlgorithm(self, parameters, context, feedback):

        source = self.parameterAsSource(parameters, self.INPUT_POLYGONS, context)
        dem_layer = self.parameterAsRasterLayer(parameters, self.INPUT_DEM, context)

        if source is None:
            raise QgsProcessingException("Invalid input polygons source.")
        if dem_layer is None or not dem_layer.isValid():
            raise QgsProcessingException("Invalid DEM raster layer.")

        slope_method = self.parameterAsEnum(parameters, self.SLOPE_METHOD, context)
        min_cov = self.parameterAsDouble(parameters, self.MIN_DEM_COV_PCT, context)
        vert_scale = self.parameterAsDouble(parameters, self.VERT_SCALE, context)

        # 1) Choose local metric CRS (global-ready)
        work_crs = self._choose_local_metric_crs(source)

        # 2) Prepare transforms
        src_crs = source.sourceCrs()
        tr_src_to_work = QgsCoordinateTransform(src_crs, work_crs, QgsProject.instance())

        # 3) Reproject DEM once to work CRS (performance)
        dem_work = self._reproject_dem_to_work_crs(dem_layer, work_crs, context, feedback)

        # 4) Warn if large extent
        self._warn_if_large_extent(source, tr_src_to_work, feedback)

        # Output fields
        out_fields = QgsFields()

        f1 = QgsField(self.F_AREA_2D, QVariant.Double)
        f1.setLength(20)
        f1.setPrecision(2)     # AREA_2D_M2 -> 2 casas
        out_fields.append(f1)

        f2 = QgsField(self.F_AREA_SURF, QVariant.Double)
        f2.setLength(20)
        f2.setPrecision(2)     # AREA_SURF_M2 -> 2 casas
        out_fields.append(f2)

        f3 = QgsField(self.F_RATIO, QVariant.Double)
        f3.setLength(20)
        f3.setPrecision(6)     # SURF_RATIO -> 6 casas
        out_fields.append(f3)

        f4 = QgsField(self.F_COV, QVariant.Double)
        f4.setLength(10)
        f4.setPrecision(1)     # DEM_COV_PCT -> 1 casa
        out_fields.append(f4)

        (sink, dest_id) = self.parameterAsSink(
            parameters,
            self.OUTPUT,
            context,
            out_fields,
            QgsWkbTypes.Polygon,
            src_crs,
        )
        if sink is None:
            raise QgsProcessingException(self.invalidSinkError(parameters, self.OUTPUT))

        total = source.featureCount()
        if total == 0:
            return {self.OUTPUT: dest_id}

        for i, f in enumerate(source.getFeatures()):
            if feedback.isCanceled():
                break

            geom_src = f.geometry()
            if geom_src is None or geom_src.isEmpty():
                continue

            # Make geometry valid for robustness
            geom_valid = geom_src.makeValid() if not geom_src.isGeosValid() else geom_src

            geom_work = QgsGeometry(geom_valid)
            geom_work.transform(tr_src_to_work)
            if geom_work.isEmpty():
                continue

            # Planar area in m² (work CRS)
            area_2d = float(geom_work.area())

            # Surface area from DEM + coverage
            area_surf, cov_pct = self._terrain_surface_area_from_dem(
                geom_work,
                dem_work,
                slope_method=slope_method,
                vert_scale=vert_scale,
            )

            ratio = float(area_surf / area_2d) if area_2d > 0 else None

            # NoData & coverage messaging (EN + PT-BR)
            if cov_pct < 100.0:
                feedback.pushInfo(
                    "Warning: DEM has missing cells inside the polygon. "
                    "Surface area computed using available data only. "
                    "(Aviso: O MDE possui células sem dados dentro do polígono. "
                    "A área foi calculada apenas com os dados disponíveis.)"
                )
            if cov_pct < min_cov:
                feedback.pushWarning(
                    f"Low DEM coverage ({cov_pct:.1f}%). Results may be unreliable. "
                    f"(Baixa cobertura do MDE ({cov_pct:.1f}%). Resultados podem ser imprecisos.)"
                )

            out_feat = QgsFeature(out_fields)
            out_feat.setGeometry(geom_valid)  # keep original CRS geometry
            out_feat.setAttributes([
                area_2d,
                float(area_surf),
                float(ratio) if ratio is not None else None,
                float(cov_pct),
            ])
            sink.addFeature(out_feat, QgsFeatureSink.FastInsert)

            feedback.setProgress(int((i + 1) * 100 / total))

        return {self.OUTPUT: dest_id}

    # --------------------------- helpers ---------------------------

    def _choose_local_metric_crs(self, source) -> QgsCoordinateReferenceSystem:
        """
        Global-ready CRS choice:
        - UTM zone from centroid lon/lat (EPSG:326xx / 327xx)
        - Polar fallback for |lat| >= 84°: EPSG:3413 (north), EPSG:3031 (south)
        """
        src_crs = source.sourceCrs()
        wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
        tr = QgsCoordinateTransform(src_crs, wgs84, QgsProject.instance())

        extent = source.sourceExtent()
        center = extent.center()
        center_wgs = tr.transform(center)

        lon = center_wgs.x()
        lat = center_wgs.y()

        if abs(lat) >= 84.0:
            epsg = 3413 if lat >= 0 else 3031
            return QgsCoordinateReferenceSystem.fromEpsgId(epsg)

        zone = int(math.floor((lon + 180.0) / 6.0) + 1)
        zone = max(1, min(60, zone))
        epsg = (32600 + zone) if lat >= 0 else (32700 + zone)
        return QgsCoordinateReferenceSystem.fromEpsgId(epsg)

    def _warn_if_large_extent(self, source, tr_src_to_work, feedback):
        try:
            ext = source.sourceExtent()
            p1 = tr_src_to_work.transform(ext.xMinimum(), ext.yMinimum())
            p2 = tr_src_to_work.transform(ext.xMaximum(), ext.yMaximum())
            w = abs(p2.x() - p1.x())
            h = abs(p2.y() - p1.y())
            if max(w, h) > 500000.0:
                feedback.pushInfo(
                    "Polygon extent is large; accuracy may decrease due to projection distortion. "
                    "(O polígono é muito extenso; a precisão pode diminuir devido à distorção cartográfica.)"
                )
        except Exception:
            pass

    def _reproject_dem_to_work_crs(self, dem_layer, work_crs, context, feedback) -> QgsRasterLayer:
        from qgis import processing

        temp_path = QgsProcessingUtils.generateTempFilename("dem_work.tif")

        feedback.pushInfo(
            "Reprojecting to local metric CRS for accurate calculation. "
            "(Reprojetando para SRC métrico local para cálculo preciso.)"
        )

        params = {
            'INPUT': dem_layer,
            'TARGET_CRS': work_crs,
            'RESAMPLING': 1,  # bilinear
            'NODATA': None,
            'TARGET_RESOLUTION': None,
            'OPTIONS': '',
            'DATA_TYPE': 0,
            'EXTRA': '',
            'OUTPUT': temp_path,
        }

        processing.run("gdal:warpreproject", params, context=context, feedback=feedback)

        dem_work = QgsRasterLayer(temp_path, "DEM_work", "gdal")
        if not dem_work.isValid():
            raise QgsProcessingException("Failed to reproject DEM to working CRS.")
        return dem_work

    def _terrain_surface_area_from_dem(self, polygon_geom_work, dem_work: QgsRasterLayer,
                                       slope_method: int, vert_scale: float):
        """
        A_surface = Σ ( A_pixel * sqrt(1 + (dz/dx)^2 + (dz/dy)^2 ) )
        """
        provider = dem_work.dataProvider()
        if provider is None:
            raise QgsProcessingException("DEM provider unavailable.")

        dem_extent = dem_work.extent()
        px = dem_work.rasterUnitsPerPixelX()
        py = dem_work.rasterUnitsPerPixelY()
        if px <= 0 or py <= 0:
            raise QgsProcessingException("Invalid DEM pixel size.")

        bbox = polygon_geom_work.boundingBox()
        win = bbox.intersect(dem_extent)
        if win.isEmpty():
            return 0.0, 0.0

        # Margin so the slope operator has neighbors
        margin_x = px
        margin_y = py
        win_exp = QgsRectangle(
            win.xMinimum() - margin_x,
            win.yMinimum() - margin_y,
            win.xMaximum() + margin_x,
            win.yMaximum() + margin_y,
        ).intersect(dem_extent)

        cols = max(3, int(math.ceil(win_exp.width() / px)))
        rows = max(3, int(math.ceil(win_exp.height() / py)))

        block = provider.block(1, win_exp, cols, rows)
        if block is None:
            raise QgsProcessingException("Failed to read DEM raster block.")

        nodata = provider.sourceNoDataValue(1)
        has_nodata = provider.sourceHasNoDataValue(1)

        Z = np.full((rows, cols), np.nan, dtype=float)
        for r in range(rows):
            for c in range(cols):
                v = block.value(r, c)
                if v is None:
                    continue
                if has_nodata and v == nodata:
                    continue
                Z[r, c] = float(v) * float(vert_scale)

        engine = QgsGeometry.createGeometryEngine(polygon_geom_work.constGet())
        engine.prepareGeometry()

        # Cell centers
        x0 = win_exp.xMinimum()
        y_top = win_exp.yMaximum()
        dx = win_exp.width() / cols
        dy = win_exp.height() / rows

        inside = np.zeros((rows, cols), dtype=bool)
        valid_dem = np.isfinite(Z)

        # FIX: QGIS 3.34 LTR expects QgsPoint here (not QgsPointXY)
        for r in range(rows):
            y = y_top - (r + 0.5) * dy
            for c in range(cols):
                x = x0 + (c + 0.5) * dx
                inside[r, c] = engine.contains(QgsPoint(x, y))

        total_cells = int(np.count_nonzero(inside))
        if total_cells == 0:
            return 0.0, 0.0

        valid_cells = int(np.count_nonzero(inside & valid_dem))
        cov_pct = (valid_cells / total_cells) * 100.0

        Zp = np.pad(Z, 1, mode='edge')

        z1 = Zp[0:rows,     0:cols]
        z2 = Zp[0:rows,     1:cols+1]
        z3 = Zp[0:rows,     2:cols+2]
        z4 = Zp[1:rows+1,   0:cols]
        z6 = Zp[1:rows+1,   2:cols+2]
        z7 = Zp[2:rows+2,   0:cols]
        z8 = Zp[2:rows+2,   1:cols+1]
        z9 = Zp[2:rows+2,   2:cols+2]

        cellx = float(dx)
        celly = float(dy)

        if slope_method == self.ZEVENBERGEN_THORNE:
            dzdx = ((z3 + z6 + z9) - (z1 + z4 + z7)) / (6.0 * cellx)
            dzdy = ((z7 + z8 + z9) - (z1 + z2 + z3)) / (6.0 * celly)
        else:
            dzdx = ((z3 + 2.0*z6 + z9) - (z1 + 2.0*z4 + z7)) / (8.0 * cellx)
            dzdy = ((z7 + 2.0*z8 + z9) - (z1 + 2.0*z2 + z3)) / (8.0 * celly)

        sf = np.sqrt(1.0 + dzdx*dzdx + dzdy*dzdy)

        pixel_area = cellx * celly
        surface_area = float(np.nansum(sf[inside & valid_dem] * pixel_area))

        return surface_area, float(cov_pct)
