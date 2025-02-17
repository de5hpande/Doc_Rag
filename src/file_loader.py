from dataclasses import dataclass
from pathlib import Path
from src.exception import CustomException
import sys
from pypdfium2 import PdfDocument
from streamlit.runtime.uploaded_file_manager import UploadedFile

from src.config import Config

TEXT_FILE_EXTENSION = ".txt"
MD_FILE_EXTENSION = ".md"

PDF_EXTENSION = ".pdf"

@dataclass
class File:
    name: str
    content: str

def extract_pdf_content(data: bytes) -> str:
    try:
        pdf = PdfDocument(data)

        content = ""
        for page in pdf:
            text_page = page.get_textpage()
            content += f"{text_page.get_text_bounded()}\n"

        return content
    except Exception as e:
        raise CustomException(e,sys)

def load_uploaded_file(uploaded_file: UploadedFile) -> File:
    try:
        file_extension = Path(uploaded_file.name).suffix

        if file_extension not in Config.ALLOWED_FILE_EXTENSIONS:
            raise ValueError(f"Invalid file extension: {file_extension} for file {uploaded_file.name}")

        if file_extension == PDF_EXTENSION:
            return File(name=uploaded_file.name, content=extract_pdf_content(uploaded_file.getvalue()))

        return File(name=uploaded_file.name, content=uploaded_file.getvalue().decode("utf-8"))
    except Exception as e:
        raise CustomException(e,sys)