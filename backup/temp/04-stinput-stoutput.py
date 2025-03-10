import os
from typing import List, Dict, Optional
import re

from langchain_google_vertexai import VertexAI
from langchain_core.runnables import chain
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field

# --- Data Models ---
class PDFContent(BaseModel):
    """Represents the content of a PDF document."""
    text: str = Field(description="The textual content extracted from the PDF.")

class Chapter(BaseModel):
    """Represents a chapter or section within a document."""
    title: str = Field(description="The title of the chapter.")
    summary: str = Field(description="A brief summary of the chapter's content.")

class PDFMetadata(BaseModel):
    """Metadata extracted from a PDF document."""
    subject: str = Field(description="The main subject or topic discussed in the PDF document.")
    chapters: List[Chapter] = Field(description="A list of chapters or sections in the PDF document.")
    author: Optional[str] = Field(description="The author of the PDF document, if available.", default=None)
    num_diagrams: int = Field(description="The approximate number of diagrams or figures in the PDF document.")

# --- Helper Functions ---
def load_pdf_content(pdf_path: str) -> PDFContent:
    """Loads the text content of a PDF document into a structured PDFContent object."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
        return PDFContent(text=text)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return PDFContent(text="")

def extract_chapter_titles(pdf_content: PDFContent) -> List[str]:
    """Extracts chapter titles from the PDF content using a simple regex."""
    # This is a VERY basic chapter title extraction.  Improve as needed.
    chapter_titles = re.findall(r"Chapter\s*\d+:\s*(.+)", pdf_content.text)
    return chapter_titles


def summarize_chapter(chapter_title: str, pdf_content: PDFContent) -> Chapter:
    """Summarizes a chapter using the LLM."""
    llm = VertexAI(model_name="gemini-pro", temperature=0.4)
    prompt = f"Summarize the content related to '{chapter_title}' from the following text: {pdf_content.text}. Provide a brief and concise summary."
    summary = llm.invoke(prompt)
    return Chapter(title=chapter_title, summary=summary)


def estimate_diagram_count(pdf_content: PDFContent) -> int:
  """Estimates the number of diagrams from the text content."""
  llm = VertexAI(model_name="gemini-pro", temperature=0.2)
  prompt = f"Estimate the number of diagrams, figures, or illustrations that are likely present in the following document text: {pdf_content.text}. Give an approximate integer count. Return only the integer count."
  diagram_count_str = llm.invoke(prompt)
  try:
      diagram_count = int(diagram_count_str.strip())
      return diagram_count
  except ValueError:
      print(f"Could not parse diagram count: {diagram_count_str}")
      return 0



def identify_subject(pdf_content: PDFContent) -> str:
    """Identifies the main subject of the PDF document."""
    llm = VertexAI(model_name="gemini-pro", temperature=0.3)
    prompt = f"Identify the primary subject or topic discussed in the following document text: {pdf_content.text}. Provide a one- or two-word answer."
    subject = llm.invoke(prompt)
    return subject.strip()

def identify_author(pdf_content: PDFContent) -> Optional[str]:
    """Identifies the author of the PDF if possible"""

    llm = VertexAI(model_name="gemini-pro", temperature=0.2)
    prompt = f"Identify the author of the document. If author is not available return 'Unknown'. Text: {pdf_content.text}"
    author = llm.invoke(prompt)
    if "Unknown" in author:
        return None
    return author

# --- Main Extraction Function ---
def extract_pdf_metadata(pdf_path: str) -> Optional[PDFMetadata]:
    """Extracts metadata from a PDF document using Langraph and structured data."""
    try:
        pdf_content = load_pdf_content(pdf_path)
        if not pdf_content.text:
            return None

        # 1. Extract Chapter Titles
        chapter_titles = extract_chapter_titles(pdf_content)

        # 2. Summarize Chapters (using LLM for each)
        chapters = [summarize_chapter(title, pdf_content) for title in chapter_titles]

        # 3. Estimate Diagram Count
        num_diagrams = estimate_diagram_count(pdf_content)

        # 4. Identify Subject
        subject = identify_subject(pdf_content)

        # 5.  Identify Author
        author = identify_author(pdf_content)

        # 6. Create PDFMetadata object
        metadata = PDFMetadata(
            subject=subject,
            chapters=chapters,
            author=author,
            num_diagrams=num_diagrams
        )
        return metadata

    except Exception as e:
        print(f"An error occurred during metadata extraction: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    pdf_file_path = "lesson.pdf"  # <--- REPLACE WITH YOUR PDF PATH

    metadata = extract_pdf_metadata(pdf_file_path)

    if metadata:
        print("PDF Metadata:")
        print(f"Subject: {metadata.subject}")
        print("Chapters:")
        for chapter in metadata.chapters:
            print(f"  - {chapter.title}: {chapter.summary}")
        print(f"Author: {metadata.author}")
        print(f"Number of Diagrams: {metadata.num_diagrams}")
    else:
        print("Failed to extract PDF metadata.")