from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.mammography.dicom_reader import read_dicom_image
from app.mammography.enums import Laterality, MammographyAnalysisStatus, MammographyView
from app.mammography.exceptions import MammographyError
from app.mammography.image_reader import read_raster_image
from app.mammography.predictor import MammographyPredictor
from app.mammography.schemas import (
    MammographyAnalyzeResponse,
    MammographyHealthResponse,
    MammographyImageMetadata,
    MammographyModelInfo,
)
from app.mammography.validation import DICOM_EXTENSIONS, MAX_UPLOAD_BYTES, validate_upload_basics

router = APIRouter(prefix="/v3/mammography", tags=["v3-mammography"])
predictor = MammographyPredictor()
UPLOAD_CHUNK_SIZE = 1024 * 1024


async def read_upload_limited(
    upload: UploadFile,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
) -> bytes:
    data = bytearray()

    while True:
        chunk = await upload.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break

        if len(data) + len(chunk) > max_upload_bytes:
            raise MammographyError(
                f"Arquivo excede o limite maximo de {max_upload_bytes} bytes.",
                status_code=413,
            )

        data.extend(chunk)

    return bytes(data)


@router.get("/health", response_model=MammographyHealthResponse, response_model_by_alias=True)
def health() -> MammographyHealthResponse:
    return MammographyHealthResponse(
        service="mammography",
        status="DEGRADED",
        model_available=predictor.is_available(),
        model_version=predictor.model_version,
    )


@router.post("/analyze", response_model=MammographyAnalyzeResponse, response_model_by_alias=True)
async def analyze(
    image: UploadFile = File(...),
    view: MammographyView = Form(...),
    laterality: Laterality = Form(...),
    report_text: str | None = Form(default=None),
) -> MammographyAnalyzeResponse:
    try:
        data = await read_upload_limited(image, MAX_UPLOAD_BYTES)
        extension = validate_upload_basics(
            filename=image.filename,
            content_type=image.content_type,
            data=data,
            max_upload_bytes=MAX_UPLOAD_BYTES,
        )

        read_result = (
            await run_in_threadpool(read_dicom_image, data, view, laterality)
            if extension in DICOM_EXTENSIONS
            else await run_in_threadpool(read_raster_image, data, extension, image.content_type)
        )
        metadata = {
            **read_result["metadata"],
            "view": view,
            "laterality": laterality,
        }

        return MammographyAnalyzeResponse(
            status=MammographyAnalysisStatus.MODEL_NOT_AVAILABLE,
            image_accepted=True,
            image_metadata=MammographyImageMetadata(**metadata),
            model=MammographyModelInfo(
                available=predictor.is_available(),
                version=predictor.model_version,
            ),
            message="Imagem validada. O modelo de mamografia ainda nao foi integrado.",
            limitations=[
                "Modelo experimental ainda nao disponivel",
                "O resultado nao representa diagnostico clinico",
            ],
        )
    except MammographyError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao processar mamografia.",
        ) from exc
