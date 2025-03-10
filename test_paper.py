import os
from typing import List, Dict, Optional, Literal
import re
import json
from  pdbwhereami import whereami

from langchain_core.runnables import chain
from langchain_core.output_parsers import PydanticOutputParser
import datetime
from models import PDFContent, MCQ, Summary, Assessment
from common import load_pdf_content_from_gcs
from langchain_google_vertexai import VertexAI
from common import clean_llm_output, save_assessment_to_json
from common import create_mcq_pdf
from langchain_core.prompts import PromptTemplate  # new import

def generate_assessment(pdf_content: PDFContent, file_uris: List[str]) -> Assessment:
    """Generates structured assessment"""
    # llm = VertexAI(model_name="gemini-pro", temperature=0.7)
    llm = VertexAI(model_name="gemini-1.5-pro-002", temperature=0.7)

    prompt_path = "prompts/02-gen-testpaper.prompt"
    with open(prompt_path, "r") as f:
        template = f.read()
        
    prompt_template = PromptTemplate(
        input_variables=["test_paper_text"], # Explicitly define input variable
        template=template,
    )


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
def generate_assessment_from_gcs(gcs_uri: str, test_paper_summary: Dict) -> Optional[Assessment]:
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
