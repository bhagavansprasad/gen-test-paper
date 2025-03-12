from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

# --- Data Models ---
class PDFContent(BaseModel):
    """Represents the content of a PDF document."""
    text: str = Field(description="The textual content extracted from the PDF.")

class MCQ(BaseModel):
    """Represents a Multiple Choice Question."""
    question: str = Field(description="The question text.")
    options: List[str] = Field(description="List of options (A, B, C, D)")
    answer: str = Field(description="The correct answer.")
    difficulty: str = Field(description="Complexity level (easy, medium, tough).")

class Summary(BaseModel):
    """Summary of the Lesson Content"""
    filenames: List[str] = Field(description="List of filenames")
    chapter_name: str = Field(description="Name of Chapter")
    sub_chapters: List[str] = Field(description="List of Sub-Chapters")
    number_of_sections: int = Field(description="number_of_sections")
    number_of_diagrams: int = Field(description="number_of_diagrams")
    # mcqs_generated: Dict[str, str] = Field(description="Number of MCQs by difficulty")
    mcqs_generated: Dict[Literal["easy", "medium", "hard"], int] = Field(description="Number of MCQs by difficulty (easy, medium, hard)")

class Assessment(BaseModel):
    """Overall Assessment"""
    file_uris: List[str] = Field(description="list of URIs")
    summary: Summary = Field(description="Summary of Assessment")
    mcqs: List[MCQ] = Field(description="Generated Mcqs")
