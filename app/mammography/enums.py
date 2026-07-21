from __future__ import annotations

from enum import Enum


class MammographyView(str, Enum):
    CC = "CC"
    MLO = "MLO"


class Laterality(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class MammographyAnalysisStatus(str, Enum):
    MODEL_NOT_AVAILABLE = "MODEL_NOT_AVAILABLE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

