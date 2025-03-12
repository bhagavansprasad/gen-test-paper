import logging  # Add this import
from typing import List, Dict, Optional, Literal
from langchain_core.prompts import PromptTemplate
from src.utils.utilities import load_pdf_content_from_gcs, load_pdf_content_local
from langchain_google_vertexai import VertexAI
from src.utils.utilities import clean_llm_output
import json
from src.models import PDFContent

logger = logging.getLogger(__name__)  # Get a logger for this module

def summarize_test_paper(test_paper_path: str) -> Optional[Dict]:
    """Summarizes a test paper using an LLM and returns JSON."""
    logger.debug(f"Entering summarize_test_paper with path: {test_paper_path}")
    try:
        # 1. Load the Prompt
        logger.debug(f"Loading prompt from: prompts/01-summarize-testpaper.prompt")
        prompt_path = "prompts/01-summarize-testpaper.prompt"
        with open(prompt_path, "r") as f:
            template = f.read()
        logger.debug("Prompt template read successfully.")

        prompt_template = PromptTemplate(
            input_variables=["test_paper_text"],
            template=template,
        )
        logger.debug("Prompt template created.")

        # 2. Load the Test Paper (PDF)
        logger.debug(f"Loading test paper content from: {test_paper_path}")
        if test_paper_path.startswith("gs://"):
            logger.debug("Loading from GCS.")
            pdf_content = load_pdf_content_from_gcs(test_paper_path)
        else:  # Assume it's a local path
            logger.debug("Loading from local path.")
            pdf_content = load_pdf_content_local(test_paper_path)

        if not pdf_content.text:
            logger.error(f"Could not load test paper content from: {test_paper_path}")
            return None
        logger.debug(f"PDF content loaded. First 100 chars: {pdf_content.text[:100]}...")

        try:
            logger.debug("Formatting prompt with test paper text.")
            formatted_prompt = prompt_template.format(test_paper_text=pdf_content.text)
            logger.debug("Prompt formatted successfully.")
        except Exception as e:
            logger.exception(f"Error formatting prompt: {e}")
            return None

        # 3. Create the LLM
        logger.debug("Creating VertexAI LLM instance.")
        llm = VertexAI(model_name="gemini-1.5-pro-002", temperature=0.2)
        logger.debug("VertexAI LLM instance created.")

        # 5. Invoke the LLM
        logger.debug("Invoking LLM with formatted prompt.")
        llm_response = llm.invoke(formatted_prompt)
        logger.debug(f"LLM response received len: {len(llm_response)}")

        # 6. Clean and Parse JSON Output
        logger.debug("Cleaning LLM output.")
        cleaned_output = clean_llm_output(llm_response)
        logger.debug(f"Cleaned LLM output len: {len(cleaned_output)}")

        try:
            logger.debug("Parsing cleaned LLM output to JSON.")
            summary_json = json.loads(cleaned_output)
            logger.info("Test paper summarized successfully.")
            # logger.debug(f"Summary JSON: {summary_json}")
            return summary_json
        except json.JSONDecodeError as e:
            logger.exception(f"JSONDecodeError: {e}")
            logger.debug(f"LLM Output (for debugging): {cleaned_output}")
            return None

    except Exception as e:
        logger.exception(f"Error summarizing test paper: {e}")
        return None
    finally:
        logger.debug("Exiting summarize_test_paper.")  # Log function exit
