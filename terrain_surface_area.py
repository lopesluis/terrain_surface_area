# -*- coding: utf-8 -*-
from qgis.core import QgsApplication
from .processing_provider.provider import Provider


class TerrainSurfaceAreaPlugin:
    """
    QGIS plugin entry point (Processing provider plugin).
    """

    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initProcessing(self):
        self.provider = Provider()
        QgsApplication.processingRegistry().addProvider(self.provider)

    def initGui(self):
        self.initProcessing()

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
