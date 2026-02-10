# -*- coding: utf-8 -*-

def classFactory(iface):
    from .terrain_surface_area import TerrainSurfaceAreaPlugin
    return TerrainSurfaceAreaPlugin(iface)
