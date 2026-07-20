from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.mammography.enums import Laterality, MammographyAnalysisStatus, MammographyView


class CamelModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class MammographyImageMetadata(CamelModel):
    width: int
    height: int
    format: str
    channels: int
    dtype: str
    view: MammographyView
    laterality: Laterality
    photometric_interpretation: Optional[str] = Field(
        default=None,
        alias="photometricInterpretation",
    )
    bits_allocated: Optional[int] = Field(default=None, alias="bitsAllocated")
    bits_stored: Optional[int] = Field(default=None, alias="bitsStored")


class MammographyModelInfo(CamelModel):
    available: bool
    version: Optional[str] = None


class MammographyAnalyzeResponse(CamelModel):
    status: MammographyAnalysisStatus
    image_accepted: bool = Field(alias="imageAccepted")
    image_metadata: MammographyImageMetadata = Field(alias="imageMetadata")
    model: MammographyModelInfo
    message: str
    limitations: List[str]


class MammographyHealthResponse(CamelModel):
    service: str
    status: str
    model_available: bool = Field(alias="modelAvailable")
    model_version: Optional[str] = Field(default=None, alias="modelVersion")

