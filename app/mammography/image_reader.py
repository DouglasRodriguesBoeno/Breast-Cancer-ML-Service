from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.mammography.exceptions import MammographyReadError, MammographyValidationError
from app.mammography.validation import (
    validate_pixel_count,
    validate_raster_format_consistency,
)


def read_raster_image(
    data: bytes,
    extension: str,
    content_type: str | None,
) -> Dict[str, Any]:
    try:
        with Image.open(BytesIO(data)) as image:
            real_format = image.format
            image.verify()

        with Image.open(BytesIO(data)) as image:
            format_name = validate_raster_format_consistency(
                extension=extension,
                content_type=content_type,
                real_format=real_format,
            )
            width, height = image.size
            validate_pixel_count(width, height)

            array = np.asarray(image)
            channels = 1 if array.ndim == 2 else int(array.shape[2])
            return {
                "array": array,
                "metadata": {
                    "width": int(width),
                    "height": int(height),
                    "format": format_name,
                    "channels": channels,
                    "dtype": str(array.dtype),
                },
            }
    except Image.DecompressionBombError as exc:
        raise MammographyValidationError(
            "Imagem excede o limite seguro de pixels para processamento.",
            status_code=413,
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MammographyReadError("Imagem PNG/JPEG invalida ou corrompida.") from exc
