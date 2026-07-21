from __future__ import annotations

from typing import Any

import numpy as np

from app.mammography.exceptions import MammographyModelNotAvailableError


def prepare_for_inference(image: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    """Placeholder for the future mammography preprocessing pipeline.

    Planned steps include MONOCHROME1 correction, useful-area crop, percentile
    normalization, resize, tensor conversion, and 1-channel to 3-channel adaption.
    """
    raise MammographyModelNotAvailableError()

