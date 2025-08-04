import PyPDF2
import pdfplumber
from docx import Document
import logging
from pathlib import Path
from typing import Optional, Dict, Any

class TextExtractor:
    """Handles text extraction from PDF and DOCX files."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_text(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF or DOCX file.
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_path.suffix.lower() == '.docx':
            return self._extract_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _extract_from_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from PDF using multiple methods for reliability."""
        text = ""
        metadata = {}
        
        try:
            # Method 1: pdfplumber (more accurate)
            with pdfplumber.open(file_path) as pdf:
                metadata = {
                    'pages': len(pdf.pages),
                    'title': pdf.metadata.get('Title', 'Unknown'),
                    'author': pdf.metadata.get('Author', 'Unknown'),
                    'subject': pdf.metadata.get('Subject', ''),
                    'file_size': file_path.stat().st_size
                }
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # Fallback: PyPDF2 if pdfplumber fails
            if not text.strip():
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
        
        except Exception as e:
            self.logger.error(f"Error extracting PDF text: {e}")
            raise
        
        return {
            'text': text.strip(),
            'metadata': metadata,
            'word_count': len(text.split()),
            'char_count': len(text)
        }
    
    def _extract_from_docx(self, file_path: Path) -> Dict[str, Any]:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
            # Extract metadata
            properties = doc.core_properties
            metadata = {
                'title': properties.title or 'Unknown',
                'author': properties.author or 'Unknown',
                'subject': properties.subject or '',
                'pages': len(doc.sections),
                'file_size': file_path.stat().st_size
            }
            
            return {
                'text': text.strip(),
                'metadata': metadata,
                'word_count': len(text.split()),
                'char_count': len(text)
            }
        
        except Exception as e:
            self.logger.error(f"Error extracting DOCX text: {e}")
            raise


def test_pdf_extraction():
    extractor = TextExtractor()
    sample_pdf = "src/sampleText.pdf"
    try:
        result = extractor.extract_text(sample_pdf)
        print("Extracted Text:\n")
        print(result['text'])
        print("\nMetadata:", result['metadata'])
        print("Word Count:", result['word_count'])
        print("Character Count:", result['char_count'])
    except Exception as e:
        print(f"Extraction failed: {e}")

if __name__ == "__main__":
    test_pdf_extraction()

