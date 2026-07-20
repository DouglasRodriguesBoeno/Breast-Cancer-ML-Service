from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import pydicom
from pydicom.errors import InvalidDicomError

from app.mammography.exceptions import MammographyReadError, MammographyValidationError
from app.mammography.validation import validate_dimensions


def read_dicom_image(data: bytes) -> Dict[str, Any]:
    try:
        dataset = pydicom.dcmread(BytesIO(data), force=False)
    except InvalidDicomError as exc:
        raise MammographyReadError("Arquivo DICOM invalido ou corrompido.") from exc
    except Exception as exc:
        raise MammographyReadError("Nao foi possivel ler o arquivo DICOM.") from exc

    if "PixelData" not in dataset:
        raise MammographyValidationError("Arquivo DICOM sem PixelData valido.")

    try:
        array = dataset.pixel_array
    except Exception as exc:
        raise MammographyReadError(
            "Nao foi possivel decodificar os pixels do DICOM. Verifique se o codec necessario esta instalado."
        ) from exc

    rows = int(getattr(dataset, "Rows", 0) or 0)
    columns = int(getattr(dataset, "Columns", 0) or 0)
    validate_dimensions(columns, rows)

    channels = 1 if array.ndim == 2 else int(array.shape[-1])
    return {
        "array": array,
        "metadata": {
            "width": columns,
            "height": rows,
            "format": "DICOM",
            "channels": channels,
            "dtype": str(array.dtype),
            "photometric_interpretation": getattr(dataset, "PhotometricInterpretation", None),
            "bits_allocated": getattr(dataset, "BitsAllocated", None),
            "bits_stored": getattr(dataset, "BitsStored", None),
        },
    }

