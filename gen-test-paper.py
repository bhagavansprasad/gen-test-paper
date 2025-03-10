import os
from typing import List, Dict, Optional, Literal
import re
import json
from  pdbwhereami import whereami

from langchain_core.runnables import chain
from langchain_core.output_parsers import PydanticOutputParser
import datetime
from models import PDFContent, MCQ, Summary, Assessment
from common import load_pdf_content_from_gcs, summarize_test_paper
from langchain_google_vertexai import VertexAI
from common import clean_llm_output, save_assessment_to_json
from common import create_mcq_pdf

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
    test_paper_path = "gs://bhagavan-pub-bucket/aignite-resources/sample-test-paper1.pdf" # GCS path for sample test paper

    # Summarize the sample test paper
    test_paper_summary = summarize_test_paper(test_paper_path)
    if test_paper_summary:
        print("\n--- Test Paper Summary ---")
        print(test_paper_summary)
    else:
        print("\n--- Failed to summarize test paper ---")

    assessment = generate_assessment_from_gcs(gcs_pdf_uri)

    if assessment:
        print("Generated Assessment:")
        print(f"File URIs: {assessment.file_uris}")
        print(f"Chapter Name: {assessment.summary.chapter_name}")
        print(f"Number of MCQs: {len(assessment.mcqs)}")
    else:
        print("Failed to generate assessment.")
