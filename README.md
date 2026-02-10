# Terrain Surface Area (DEM)

Computes planar area and terrain surface area of polygons using a DEM.  
(Calcula a área plana e a área de superfície do terreno de polígonos usando um MDE.)

## Inputs
- Input polygons (camada poligonal)
- Elevation raster (DEM) (raster MDE)

## Outputs (fields)
- AREA_2D_M2 — Planar area in m² (Área plana em m²)
- AREA_SURF_M2 — Terrain surface area in m² (Área de superfície do terreno em m²)
- SURF_RATIO — AREA_SURF_M2 / AREA_2D_M2 (Relação área superfície / área plana)
- DEM_COV_PCT — Valid DEM coverage inside polygon (%) (Cobertura válida do MDE no polígono)

## Method
Surface area is computed per raster cell using a slope-based correction:

A_surface = Σ ( A_pixel * sqrt(1 + (dz/dx)^2 + (dz/dy)^2 ) )

Slope operators:
- Horn (default) — robust for most DEMs (robusto para a maioria dos MDEs)
- Zevenbergen & Thorne — smoother, may be more sensitive to noise (mais suave, pode ser sensível a ruído)

## Global CRS handling
The plugin automatically reprojects computations to a local metric CRS:
- UTM zone derived from polygon centroid (EPSG:326xx/327xx)
- Polar fallback for |lat| ≥ 84° (EPSG:3413 north / EPSG:3031 south)

(O cálculo é reprojetado automaticamente para SRC métrico local.)

## NoData handling
If DEM has NoData inside polygons, results are computed using available cells only and DEM_COV_PCT reports coverage.
(Se houver NoData, o cálculo usa apenas células válidas e informa a cobertura.)
