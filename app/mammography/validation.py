from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from app.mammography.exceptions import MammographyValidationError

MAX_UPLOAD_BYTES = int(os.getenv("MAMMOGRAPHY_MAX_UPLOAD_BYTES", str(30 * 1024 * 1024)))

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DICOM_EXTENSIONS = {".dcm"}

ALLOWED_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "application/dicom",
    "application/octet-stream",
    "application/x-dicom",
}

RASTER_FORMAT_BY_EXTENSION = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
}

RASTER_MIME_BY_FORMAT = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
}


def extension_from_filename(filename: Optional[str]) -> str:
    if not filename:
        raise MammographyValidationError("Nome do arquivo ausente.")
    return Path(filename).suffix.lower()


def validate_upload_basics(
    *,
    filename: Optional[str],
    content_type: Optional[str],
    data: bytes,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> str:
    if not data:
        raise MammographyValidationError("Arquivo vazio.")

    if len(data) > max_upload_bytes:
        raise MammographyValidationError(
            f"Arquivo excede o limite maximo de {max_upload_bytes} bytes.",
            status_code=413,
        )

    extension = extension_from_filename(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise MammographyValidationError("Extensao de arquivo nao permitida.")

    if content_type and content_type not in ALLOWED_MIME_TYPES:
        raise MammographyValidationError("MIME type nao permitido.")

    return extension


def validate_dimensions(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise MammographyValidationError("Imagem com dimensoes invalidas.")


def validate_raster_format_consistency(
    *,
    extension: str,
    content_type: Optional[str],
    real_format: Optional[str],
) -> str:
    normalized_format = (real_format or "").upper()
    expected_format = RASTER_FORMAT_BY_EXTENSION.get(extension)

    if normalized_format not in RASTER_MIME_BY_FORMAT:
        raise MammographyValidationError("Formato real da imagem nao permitido.")

    if expected_format != normalized_format:
        raise MammographyValidationError(
            "Formato real da imagem diverge da extensao informada."
        )

    expected_mime = RASTER_MIME_BY_FORMAT[normalized_format]
    if content_type and content_type != expected_mime:
        raise MammographyValidationError(
            "MIME type diverge do formato real da imagem."
        )

    return normalized_format
