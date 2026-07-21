from __future__ import annotations

from typing import Optional

import numpy as np

from app.mammography.exceptions import MammographyModelNotAvailableError


class MammographyPredictor:
    @property
    def model_version(self) -> Optional[str]:
        return None

    def is_available(self) -> bool:
        return False

    def predict(self, image: np.ndarray) -> dict:
        raise MammographyModelNotAvailableError()

