from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from app.main import app

client = TestClient(app)


def make_image_bytes(format_name: str = "PNG") -> bytes:
    image = Image.fromarray(np.zeros((12, 10), dtype=np.uint8))
    buffer = BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()


def make_dicom_bytes(include_pixel_data: bool = True) -> bytes:
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6"
    file_meta.ImplementationClassUID = "1.2.3.4"

    dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False
    dataset.SOPClassUID = file_meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.Modality = "MG"
    dataset.Rows = 8
    dataset.Columns = 7
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0

    if include_pixel_data:
        dataset.PixelData = np.zeros((8, 7), dtype=np.uint16).tobytes()

    buffer = BytesIO()
    dataset.save_as(buffer, write_like_original=False)
    return buffer.getvalue()


def post_image(
    data: bytes,
    filename: str,
    content_type: str,
    view: str = "MLO",
    laterality: str = "LEFT",
):
    return client.post(
        "/v3/mammography/analyze",
        files={"image": (filename, data, content_type)},
        data={"view": view, "laterality": laterality},
    )


def test_mammography_health() -> None:
    response = client.get("/v3/mammography/health")

    assert response.status_code == 200
    assert response.json() == {
        "service": "mammography",
        "status": "DEGRADED",
        "modelAvailable": False,
        "modelVersion": None,
    }


@pytest.mark.parametrize(
    ("format_name", "filename", "content_type", "expected_format"),
    [
        ("PNG", "image.png", "image/png", "PNG"),
        ("JPEG", "image.jpg", "image/jpeg", "JPEG"),
    ],
)
def test_analyze_valid_raster_images(
    format_name: str,
    filename: str,
    content_type: str,
    expected_format: str,
) -> None:
    response = post_image(make_image_bytes(format_name), filename, content_type)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "MODEL_NOT_AVAILABLE"
    assert body["imageAccepted"] is True
    assert body["imageMetadata"]["format"] == expected_format
    assert body["imageMetadata"]["view"] == "MLO"
    assert body["imageMetadata"]["laterality"] == "LEFT"
    assert body["model"] == {"available": False, "version": None}


def test_empty_file_is_rejected() -> None:
    response = post_image(b"", "empty.png", "image/png")

    assert response.status_code == 400
    assert "Arquivo vazio" in response.json()["detail"]


def test_extension_not_allowed() -> None:
    response = post_image(b"abc", "image.gif", "image/gif")

    assert response.status_code == 400
    assert "Extensao" in response.json()["detail"]


def test_corrupted_image_is_rejected() -> None:
    response = post_image(b"not an image", "image.png", "image/png")

    assert response.status_code == 400
    assert "corrompida" in response.json()["detail"]


def test_invalid_view_is_rejected() -> None:
    response = post_image(make_image_bytes(), "image.png", "image/png", view="LM")

    assert response.status_code == 422


def test_invalid_laterality_is_rejected() -> None:
    response = post_image(make_image_bytes(), "image.png", "image/png", laterality="BOTH")

    assert response.status_code == 422


def test_file_above_limit_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.api import mammography_routes

    monkeypatch.setattr(mammography_routes, "MAX_UPLOAD_BYTES", 4, raising=False)
    response = post_image(make_image_bytes(), "image.png", "image/png")

    assert response.status_code == 413


def test_valid_synthetic_dicom() -> None:
    response = post_image(make_dicom_bytes(), "image.dcm", "application/dicom")

    assert response.status_code == 200
    metadata = response.json()["imageMetadata"]
    assert metadata["format"] == "DICOM"
    assert metadata["width"] == 7
    assert metadata["height"] == 8
    assert metadata["channels"] == 1
    assert metadata["dtype"] == "uint16"
    assert metadata["photometricInterpretation"] == "MONOCHROME2"
    assert metadata["bitsAllocated"] == 16
    assert metadata["bitsStored"] == 12
    assert "PatientName" not in metadata
    assert "PatientID" not in metadata


def test_dicom_without_pixel_data_is_rejected() -> None:
    response = post_image(make_dicom_bytes(include_pixel_data=False), "image.dcm", "application/dicom")

    assert response.status_code == 400
    assert "PixelData" in response.json()["detail"]


def test_response_has_no_fake_scores_or_diagnosis() -> None:
    response = post_image(make_image_bytes(), "image.png", "image/png")

    assert response.status_code == 200
    body = response.json()
    forbidden_keys = {
        "score",
        "probability",
        "probabilityMalignant",
        "probabilityBenign",
        "diagnosis",
        "birads",
        "heatmap",
    }
    assert forbidden_keys.isdisjoint(body.keys())
    assert body["status"] == "MODEL_NOT_AVAILABLE"

