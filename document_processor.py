import os
import pandas as pd
import pytesseract
from PIL import Image
import PyPDF2
from typing import List, Dict
import hashlib

class DocumentProcessor:
    def __init__(self):
        self.supported_formats = ['.pdf', '.csv', '.txt', '.png', '.jpg', '.jpeg']
    
    def process_document(self, file_path: str, filename: str) -> List[Dict]:
        """Process a document and return chunks of text with metadata."""
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Extract text based on file type
        if file_extension == '.pdf':
            text = self._extract_pdf_text(file_path)
        elif file_extension == '.csv':
            text = self._extract_csv_text(file_path)
        elif file_extension == '.txt':
            text = self._extract_txt_text(file_path)
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            text = self._extract_image_text(file_path)
        else:
            raise ValueError(f"No processor available for {file_extension}")
        
        # Split text into chunks
        chunks = self._split_text_into_chunks(text, chunk_size=1000, overlap=200)
        
        # Create document chunks with metadata
        document_chunks = []
        for i, chunk in enumerate(chunks):
            document_chunks.append({
                'content': chunk,
                'metadata': {
                    'filename': filename,
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'file_type': file_extension,
                    'document_id': self._generate_document_id(filename)
                }
            })
        
        return document_chunks

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF, fallback to OCR if extraction fails or is too short."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")

        # Fallback to OCR if text is too short
        if len(text.strip()) < 100:
            try:
                print("🔁 Falling back to OCR for PDF...")
                from pdf2image import convert_from_path
                images = convert_from_path(file_path)
                for img in images:
                    text += pytesseract.image_to_string(img)
            except Exception as ocr_error:
                raise Exception(f"OCR fallback failed: {str(ocr_error)}")

        return text
    
    def _extract_csv_text(self, file_path: str) -> str:
        """Extract text from CSV file."""
        try:
            df = pd.read_csv(file_path)
            text = f"Dataset Information:\n"
            text += f"Columns: {', '.join(df.columns.tolist())}\n"
            text += f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n"
            
            text += "Column Descriptions:\n"
            for col in df.columns:
                text += f"- {col}: {df[col].dtype}, "
                if df[col].dtype == 'object':
                    unique_vals = df[col].nunique()
                    text += f"{unique_vals} unique values"
                else:
                    text += f"range: {df[col].min()} to {df[col].max()}"
                text += "\n"
            
            text += f"\nSample Data (first 5 rows):\n"
            text += df.head().to_string()
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                text += f"\n\nStatistical Summary:\n"
                text += df[numeric_cols].describe().to_string()
            
            return text
        except Exception as e:
            raise Exception(f"Error extracting CSV text: {str(e)}")
    
    def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error extracting TXT text: {str(e)}")
    
    def _extract_image_text(self, file_path: str) -> str:
        """Extract text from image using OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            raise Exception(f"Error extracting image text: {str(e)}")
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                last_space = text.rfind(' ', start, end)
                
                if last_period > start + chunk_size * 0.7:
                    end = last_period + 1
                elif last_newline > start + chunk_size * 0.7:
                    end = last_newline
                elif last_space > start + chunk_size * 0.7:
                    end = last_space
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _generate_document_id(self, filename: str) -> str:
        """Generate a unique document ID based on filename."""
        return hashlib.md5(filename.encode()).hexdigest()[:8]
