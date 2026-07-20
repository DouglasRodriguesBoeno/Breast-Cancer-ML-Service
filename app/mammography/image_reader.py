from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.mammography.exceptions import MammographyReadError
from app.mammography.validation import validate_dimensions


def read_raster_image(data: bytes, extension: str) -> Dict[str, Any]:
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()

        with Image.open(BytesIO(data)) as image:
            array = np.asarray(image)
            width, height = image.size
            validate_dimensions(width, height)

            channels = 1 if array.ndim == 2 else int(array.shape[2])
            return {
                "array": array,
                "metadata": {
                    "width": int(width),
                    "height": int(height),
                    "format": "JPEG" if extension in {".jpg", ".jpeg"} else "PNG",
                    "channels": channels,
                    "dtype": str(array.dtype),
                },
            }
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MammographyReadError("Imagem PNG/JPEG invalida ou corrompida.") from exc

