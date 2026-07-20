from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

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
        data = await image.read()
        extension = validate_upload_basics(
            filename=image.filename,
            content_type=image.content_type,
            data=data,
            max_upload_bytes=MAX_UPLOAD_BYTES,
        )

        read_result = (
            read_dicom_image(data)
            if extension in DICOM_EXTENSIONS
            else read_raster_image(data, extension)
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
