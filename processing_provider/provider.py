# -*- coding: utf-8 -*-
from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from .terrain_surface_area_algorithm import TerrainSurfaceAreaAlgorithm


class Provider(QgsProcessingProvider):

    def loadAlgorithms(self):
        self.addAlgorithm(TerrainSurfaceAreaAlgorithm())

    def id(self) -> str:
        # unique, stable, not localized
        return 'terrain_surface_area'

    def name(self) -> str:
        return self.tr('Terrain Surface Area (DEM)')

    def icon(self) -> QIcon:
        return QgsProcessingProvider.icon(self)
