from pathlib import Path
import pydicom 
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError
from app.mammography.validation import validate_pixel_count

def load_dicom(path: Path) -> Dataset:
    try:
        return pydicom.dcmread(path)
    except InvalidDicomError as exc:
        raise ValueError(f"Invalid DICOM file: {path}") from exc
    
def validate_basic_dicom(dataset: Dataset) -> None:
    modality = getattr(dataset, "Modality", None)
    rows = getattr(dataset, "Rows", None)
    columns = getattr(dataset, "Columns", None)
    number_of_frames = getattr(dataset, "NumberOfFrames", None)

    if not modality:
        raise ValueError("DICOM file is missing Modality attribute.")

    if str(modality).strip().upper() != "MG":
        raise ValueError("DICOM file is not a mammography exam.")

    if not rows:
        raise ValueError("DICOM file is missing Rows attribute.")

    if not columns:
        raise ValueError("DICOM file is missing Columns attribute.")

    if "PixelData" not in dataset:
        raise ValueError("DICOM file is missing PixelData attribute.")

    if number_of_frames is not None:
        try:
            number_of_frames = int(number_of_frames)
        except (TypeError, ValueError) as exc:
            raise ValueError("DICOM file has an invalid NumberOfFrames attribute.") from exc

        if number_of_frames <= 0:
            raise ValueError("DICOM file has an invalid NumberOfFrames attribute.")

    validate_pixel_count(columns, rows)

def inspect_pixels(dataset: Dataset) -> None:
    photometric_interpretation = getattr(
        dataset,
        "PhotometricInterpretation",
        None,
    )

    pixels = dataset.pixel_array
    bits_allocated = getattr(dataset, "BitsAllocated", None)
    bits_stored = getattr(dataset, "BitsStored", None)

    print(f"Photometric Interpretation: {photometric_interpretation}")
    print(f"Shape: {pixels.shape}")
    print(f"Dtype: {pixels.dtype}")
    print(f"Min pixel value: {pixels.min()}")
    print(f"Max pixel value: {pixels.max()}")    
    print(f"Bits Allocated: {bits_allocated}")
    print(f"Bits Stored: {bits_stored}")