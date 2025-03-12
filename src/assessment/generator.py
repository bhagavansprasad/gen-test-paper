import logging  # Add this import
import os
from typing import List, Dict, Optional, Literal
import re
import json

from langchain_core.runnables import chain
from langchain_core.output_parsers import PydanticOutputParser
import datetime
from src.models import PDFContent, MCQ, Summary, Assessment
from src.utils.utilities import load_pdf_content_from_gcs
from langchain_google_vertexai import VertexAI
from src.utils.utilities import clean_llm_output, save_assessment_to_json
from langchain_core.prompts import PromptTemplate  # new import

logger = logging.getLogger(__name__)  # Get a logger for this module

def generate_assessment(pdf_content: PDFContent, test_paper_summary: Dict, file_uris: List[str]) -> Assessment:
    """Generates structured assessment"""
    logger.debug("Entering generate_assessment")

    logger.debug("Initializing VertexAI LLM.")
    # llm = VertexAI(model_name="gemini-pro", temperature=0.7)
    llm = VertexAI(model_name="gemini-1.5-pro-002", temperature=0.7)
    logger.debug("VertexAI LLM initialized.")

    logger.debug(f"Loading prompt from: prompts/02-gen-testpaper.prompt")
    prompt_path = "prompts/02-gen-testpaper.prompt"
    with open(prompt_path, "r") as f:
        template = f.read()
    prompt_template = PromptTemplate(template=template)
    logger.debug("Prompt template loaded")


    try:
        logger.debug("Formatting prompt with chapter content summary and test paper analysis.")
        formatted_prompt = prompt_template.format(
            chapter_content_summary=pdf_content.text,
            test_paper_analysis=json.dumps(test_paper_summary)
        )
        logger.debug("Prompt formatted successfully.")
    except Exception as e:
        logger.exception(f"Error formatting prompt: {e}")
        return None

    logger.debug("Invoking LLM with formatted prompt.")
    llm_response = llm.invoke(formatted_prompt)
    logger.debug(f"LLM response received len: {len(llm_response)}")
    logger.info(f"\n---LLM response--")
    logger.info(llm_response)

    llm_output = llm_response.strip()
    logger.debug("LLM output received.")

    if not llm_output:
        logger.error(f"Failed to extract details from document: {file_uris}")
        return None

    logger.debug(f"Type of LLM output: {type(llm_output)}")
    # Add more if you intend to handle different types

    logger.debug("Exiting generate_assessment.")
    return llm_output
