from __future__ import annotations

from io import BytesIO
from typing import Any, Dict

import pydicom
from pydicom.errors import InvalidDicomError

from app.mammography.enums import Laterality, MammographyView
from app.mammography.exceptions import MammographyReadError, MammographyValidationError
from app.mammography.validation import validate_dimensions


LATERALITY_MAP = {
    "L": Laterality.LEFT,
    "LEFT": Laterality.LEFT,
    "R": Laterality.RIGHT,
    "RIGHT": Laterality.RIGHT,
}


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_view(value: Any) -> MammographyView | None:
    text = _optional_text(value)
    if not text:
        return None
    try:
        return MammographyView(text.upper())
    except ValueError as exc:
        raise MammographyValidationError(
            "ViewPosition do DICOM nao e suportado nesta versao.",
            status_code=422,
        ) from exc


def _normalize_laterality(value: Any) -> Laterality | None:
    text = _optional_text(value)
    if not text:
        return None
    laterality = LATERALITY_MAP.get(text.upper())
    if laterality is None:
        raise MammographyValidationError(
            "Lateralidade do DICOM nao e suportada nesta versao.",
            status_code=422,
        )
    return laterality


def _validate_dicom_context(
    *,
    view_position: MammographyView | None,
    image_laterality: Laterality | None,
    expected_view: MammographyView,
    expected_laterality: Laterality,
) -> None:
    if view_position is not None and view_position != expected_view:
        raise MammographyValidationError(
            "ViewPosition do DICOM diverge do campo view enviado.",
            status_code=422,
        )

    if image_laterality is not None and image_laterality != expected_laterality:
        raise MammographyValidationError(
            "Lateralidade do DICOM diverge do campo laterality enviado.",
            status_code=422,
        )


def read_dicom_image(
    data: bytes,
    expected_view: MammographyView,
    expected_laterality: Laterality,
) -> Dict[str, Any]:
    try:
        dataset = pydicom.dcmread(BytesIO(data), force=False)
    except InvalidDicomError as exc:
        raise MammographyReadError("Arquivo DICOM invalido ou corrompido.") from exc
    except Exception as exc:
        raise MammographyReadError("Nao foi possivel ler o arquivo DICOM.") from exc

    if "PixelData" not in dataset:
        raise MammographyValidationError("Arquivo DICOM sem PixelData valido.")

    modality = _optional_text(getattr(dataset, "Modality", None))
    if (modality or "").upper() != "MG":
        raise MammographyValidationError(
            "Apenas DICOM de mamografia com Modality MG e aceito nesta versao.",
            status_code=422,
        )

    number_of_frames_raw = getattr(dataset, "NumberOfFrames", None)
    if number_of_frames_raw is not None:
        try:
            number_of_frames = int(number_of_frames_raw)
        except (TypeError, ValueError) as exc:
            raise MammographyValidationError(
                "NumberOfFrames do DICOM e invalido.",
                status_code=422,
            ) from exc
        if number_of_frames > 1:
            raise MammographyValidationError(
                "DICOM multiframe nao e aceito nesta versao.",
                status_code=422,
            )

    view_position = _normalize_view(getattr(dataset, "ViewPosition", None))
    image_laterality = _normalize_laterality(
        getattr(dataset, "ImageLaterality", None) or getattr(dataset, "Laterality", None)
    )
    _validate_dicom_context(
        view_position=view_position,
        image_laterality=image_laterality,
        expected_view=expected_view,
        expected_laterality=expected_laterality,
    )

    try:
        array = dataset.pixel_array
    except Exception as exc:
        raise MammographyReadError(
            "Nao foi possivel decodificar os pixels do DICOM. Verifique se o codec necessario esta instalado."
        ) from exc

    rows = int(getattr(dataset, "Rows", 0) or 0)
    columns = int(getattr(dataset, "Columns", 0) or 0)
    validate_dimensions(columns, rows)

    if array.ndim != 2:
        raise MammographyValidationError(
            "Apenas DICOM 2D de frame unico e aceito nesta versao.",
            status_code=422,
        )

    return {
        "array": array,
        "metadata": {
            "width": columns,
            "height": rows,
            "format": "DICOM",
            "channels": 1,
            "dtype": str(array.dtype),
            "photometric_interpretation": getattr(dataset, "PhotometricInterpretation", None),
            "bits_allocated": getattr(dataset, "BitsAllocated", None),
            "bits_stored": getattr(dataset, "BitsStored", None),
            "view_position": view_position.value if view_position else None,
            "image_laterality": image_laterality.value if image_laterality else None,
        },
    }
