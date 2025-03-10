import os
from typing import List, Dict, Optional, Literal
import re
from urllib.parse import urlparse
import json
from  pdbwhereami import whereami

from langchain_google_vertexai import VertexAI
from langchain_core.runnables import chain
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_community import GCSFileLoader
from pydantic import BaseModel, Field
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer #Add new
from reportlab.lib.styles import getSampleStyleSheet # New Import
from reportlab.lib.units import inch #added new
from reportlab.lib.styles import ParagraphStyle #added new
from reportlab.lib import colors #new dep

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


# --- Helper Functions ---
def load_pdf_content_from_gcs(gcs_uri: str) -> PDFContent:
    """Loads the text content of a PDF document from Google Cloud Storage into a structured PDFContent object."""
    try:
        parsed_uri = urlparse(gcs_uri)
        bucket_name = parsed_uri.netloc
        blob_name = parsed_uri.path.lstrip("/")

        loader = GCSFileLoader(bucket=bucket_name, blob=blob_name, project_name="bhagavangenai-444212")
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
        return PDFContent(text=text)
    except Exception as e:
        print(f"Error loading PDF from GCS: {e}")
        return PDFContent(text="")

def clean_llm_output(llm_output: str) -> str:
    """
    Cleans the LLM output by removing any leading/trailing whitespace and ```json blocks.
    """
    llm_output = llm_output.strip()
    whereami(f"LLM Response: ")
    # Remove ```json and ``` if present
    if llm_output.startswith("```json"):
        llm_output = llm_output[len("```json"):].strip()
    if llm_output.endswith("```"):
        llm_output = llm_output[:-len("```")].strip()
    whereami(f"LLM Response: ")
    return llm_output

def save_assessment_to_json(assessment: Assessment, filename: str):
    """Saves the assessment data to a JSON file."""
    try:
        with open(filename, "w") as f:
            json.dump(assessment.dict(), f, indent=2)
        print(f"Assessment saved to {filename}")
    except Exception as e:
        print(f"Error saving assessment to JSON: {e}")

def create_mcq_pdf(assessment: Assessment, filename: str):
    """Creates a PDF document containing the MCQs from the assessment, handling multi-page content."""
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()

    # Custom paragraph style for MCQs
    mcq_style = ParagraphStyle(
        name='MCQStyle',
        parent=styles['Normal'],
        fontSize=12,
        leading=14,
        spaceAfter=12,  # Add space after each MCQ
    )

    # Title style
    title_style = ParagraphStyle(
        name='TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        spaceAfter=24,
        alignment=1,  # Center alignment
    )

    # Header information style
    header_style = ParagraphStyle(
        name='HeaderStyle',
        parent=styles['Normal'],
        fontSize=12,
        leading=14,
        spaceAfter=12,
    )

    story = []

    # Header information
    header_text = f"""
        <b>Assessment Test</b><br/>
        <b>Subject:</b> {assessment.summary.chapter_name}<br/>
        <b>Sub-chapters:</b> {", ".join([sc for sc in assessment.summary.sub_chapters if sc.lower() not in ["summary", "introduction"]])}<br/>
        <b>Question types:</b> Easy: {assessment.summary.mcqs_generated["easy"]}, Medium: {assessment.summary.mcqs_generated["medium"]}, Hard: {assessment.summary.mcqs_generated["hard"]}
    """
    story.append(Paragraph("Assessment Test", title_style)) #add title
    story.append(Spacer(1, 12)) #spacer after
    story.append(Paragraph(header_text, header_style))#adding all the headers
    story.append(Spacer(1, 12))  # Add a blank line

    # Build the mcqs
    def build_mcq_paragraph(mcq: MCQ, index: int) -> Paragraph:
        text = f"{index + 1}. {mcq.question}<br/>"
        options = ["a", "b", "c", "d"]
        for i, option in enumerate(mcq.options):
            text += f"  {options[i]}. {option}<br/>"
        return Paragraph(text, mcq_style)

    # Create the story (list of flowables)
    for i, mcq in enumerate(assessment.mcqs):
        story.append(build_mcq_paragraph(mcq, i))
        story.append(Spacer(1, 12)) #space after paragraphs

    # Build the PDF
    doc.build(story)
    print(f"MCQs saved to {filename}")
        
def generate_assessment(pdf_content: PDFContent, file_uris: List[str]) -> Assessment:
    """Generates structured assessment"""
    # llm = VertexAI(model_name="gemini-pro", temperature=0.7)
    llm = VertexAI(model_name="gemini-1.5-pro-002", temperature=0.7)

    prompt = f"""
    You are an AI-powered assessment creator.

    ### **Task**
    1.  Read and analyze the provided lesson content **strictly from the given files** referenced in `file_uris`.
    2.  Identify key **concepts, definitions, formulas, and facts**.
    3.  Before generating MCQs, **summarize** the lesson content by extracting:
        - **Input Filenames**: List of the provided file URIs.
        - **Chapter Name**: Extract the main chapter name from the content.
        - **Sub-Chapters**: Identify and list sub-chapters if available.
        - **Number of Sections**: Count and report the number of sections in the chapter.
        - **Number of Diagrams**: Count and report any diagrams present in the text.
        - **Number of MCQs Generated**: Provide a count of MCQs categorized by difficulty:
            - Easy: X
            - Medium: Y
            - Hard: Z

    4.  Generate **30 multiple-choice questions (MCQs)** covering:
        - **Easy** (basic recall questions)
        - **Medium** (conceptual understanding)
        - **Hard** (application-based reasoning)

    5.  DO NOT generate vague, trivial, or overly broad questions such as:
        - "What is discussed in this chapter?"
        - "What is the main topic of this text?"
        - "Summarize the content of this lesson."
        - Any question that can be answered without reading the text.

    ### **Strict Content Restriction**
    ✅ **Use ONLY the information extracted from the given files.**
    ✅ **Do NOT generate questions requiring external knowledge.**
    ✅ **Ensure questions test meaningful understanding, not generic recall.**
    ✅ **Avoid questions that are too specific or too broad.**
    
    ### **Question Guidelines**
    - Each MCQ should have **four answer choices**.
    - The correct answer must be **clear and unambiguous**.
    - Avoid repetitive phrasing or redundant concepts.

    ### **Input**
    - **Lesson Content**: {pdf_content.text}
    - **File URIs**: {file_uris}
    - **Sub-Chapters**: (Not applicable, extracting from content)

    ### **Output Format (Plain JSON)**
    ```json
    {{
      "file_uris": {str(file_uris)},
      "summary": {{
        "filenames": [{str(file_uris)}],
        "chapter_name": "...",
        "sub_chapters": [],
        "number_of_sections": "...",
        "number_of_diagrams": "...",
        "mcqs_generated": {{
          "easy": "...",
          "medium": "...",
          "hard": "..."
        }}
      }},
      "mcqs": [
        {{
          "question": "...",
          "options": ["...", "...", "...", "..."],
          "answer": "...",
          "difficulty": "..."
        }}
        // more mcqs
      ]
    }}
    ```
    """
    llm_response = llm.invoke(prompt)

    whereami(f"LLM Response: {llm_response}")
    
    llm_output = llm_response.strip()
    whereami(f"LLM Response:")

    if not llm_output:
        print(f"Failed to extract details from document: {file_uris}")
        return None

    whereami(f"LLM Response:")

    try:
        llm_output = clean_llm_output(llm_output)
        whereami(f"LLM Response: {llm_output}")
    except Exception as e:
        print(f"Error cleaning LLM output: {e}")

    # json_file = "mcqs.json"
    # with open(json_file, "r") as f:
    #     llm_output = f.read()
            
    whereami(f"type :{type(llm_output)}")
    whereami(f"LLM Response: {llm_output}")
    try:
        assessment = json.loads(llm_output)

        #Convert Assessment to the Pydantic Model
        assessment_obj = Assessment(
            file_uris=assessment.get("file_uris", []),
            summary=Summary(
                filenames=assessment["summary"].get("filenames", []),
                chapter_name=assessment["summary"].get("chapter_name", ""),
                sub_chapters=assessment["summary"].get("sub_chapters", []),
                number_of_sections=assessment["summary"].get("number_of_sections", ""),
                number_of_diagrams=assessment["summary"].get("number_of_diagrams", ""),
                mcqs_generated=assessment["summary"].get("mcqs_generated", {})
            ),
            mcqs=[MCQ(**mcq_data) for mcq_data in assessment.get("mcqs", [])]
        )

        return assessment_obj

    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
    except Exception as e:
        print(f"Error parsing MCQ: {e}\nLLM Response: {llm_output}")
        return None


# --- Main Function ---
def generate_assessment_from_gcs(gcs_uri: str) -> Optional[Assessment]:
    """Generates a structured assessment from a PDF document stored in Google Cloud Storage."""
    try:
        pdf_content = load_pdf_content_from_gcs(gcs_uri)
        if not pdf_content.text:
            return None

        file_uris = [gcs_uri]
        assessment = generate_assessment(pdf_content, file_uris)

        #Generate filename
        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y%m%d-%H%M%S")
        base_filename = f"mcqs-{date_time_str}"

        if assessment:
            save_assessment_to_json(assessment, f"{base_filename}.json")
            create_mcq_pdf(assessment, f"{base_filename}.pdf")

        return assessment

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    gcs_pdf_uri = "gs://bhagavan-pub-bucket/aignite-resources/jemh1a1.pdf"  # Replace with your GCS URI

    assessment = generate_assessment_from_gcs(gcs_pdf_uri)

    if assessment:
        print("Generated Assessment:")
        print(f"File URIs: {assessment.file_uris}")
        print(f"Chapter Name: {assessment.summary.chapter_name}")
        print(f"Number of MCQs: {len(assessment.mcqs)}")
    else:
        print("Failed to generate assessment.")


