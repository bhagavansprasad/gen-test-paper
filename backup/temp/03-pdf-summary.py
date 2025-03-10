import os
from typing import List, Dict, Optional
import re

from langchain_google_vertexai import VertexAI
from langchain_core.runnables import chain
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field


class PDFMetadata(BaseModel):
    """Metadata extracted from a PDF document."""

    subject: str = Field(description="The main subject or topic discussed in the PDF document.")
    num_chapters: int = Field(description="The number of chapters or sections in the PDF document.")
    author: Optional[str] = Field(description="The author of the PDF document, if available.", default=None)
    num_diagrams: int = Field(description="The approximate number of diagrams or figures in the PDF document.")


def load_pdf_content(pdf_path: str) -> str:
    """Loads the text content of a PDF document."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
        return text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return ""


def analyze_pdf_content(pdf_content: str) -> str:
    """Transforms the PDF content into a prompt for analysis."""
    prompt = f"""
    Analyze the following text extracted from a PDF document and extract the following information:

    - The main subject or topic discussed in the document.
    - The number of chapters or distinct sections in the document (estimate if not explicitly stated).
    - The author of the document (if available). If author is not available return 'Unknown'.
    - The approximate number of diagrams, figures, or illustrations in the document (estimate based on mentions or visual cues).

    Text:
    {pdf_content}

    Provide your response as a JSON object with the keys "subject", "num_chapters", "author", and "num_diagrams".  Ensure "num_chapters" and "num_diagrams" are integers. If the author is unknown, set the author field to "Unknown". Return ONLY the JSON.
    """
    return prompt


# Main function to extract metadata
def extract_pdf_metadata(pdf_path: str) -> Optional[PDFMetadata]:
    """Extracts metadata from a PDF document using Langraph and a structured output."""

    try:
        pdf_content = load_pdf_content(pdf_path)
        if not pdf_content:
            return None  # Handle PDF loading failure

        llm = VertexAI(model_name="gemini-pro", temperature=0.3) #Adjust temperature as needed
        parser = PydanticOutputParser(pydantic_object=PDFMetadata)

        chain = (
            analyze_pdf_content  # Step 1: Create the prompt
            | llm  # Step 2: Pass it to the LLM
            | parser  # Step 3: Parse the output
        )

        metadata: PDFMetadata = chain.invoke(pdf_content)
        return metadata

    except Exception as e:
        print(f"An error occurred during metadata extraction: {e}")
        return None


# Example usage:
if __name__ == "__main__":
    pdf_file_path = "lesson.pdf" 

    metadata = extract_pdf_metadata(pdf_file_path)

    if metadata:
        print("PDF Metadata:")
        print(f"Subject: {metadata.subject}")
        print(f"Number of Chapters: {metadata.num_chapters}")
        print(f"Author: {metadata.author}")
        print(f"Number of Diagrams: {metadata.num_diagrams}")
    else:
        print("Failed to extract PDF metadata.")