import os
import datetime
import logging  # Add this import
from src.models import PDFContent, Assessment
from urllib.parse import urlparse
from langchain_google_community import GCSFileLoader
import json
from langchain_community.document_loaders import UnstructuredPDFLoader

logger = logging.getLogger(__name__)  # Get logger for this module.

class PDFLoadError(Exception):
    """Custom exception for PDF loading errors."""
    pass

# --- Helper Functions ---
def load_pdf_content_from_gcs(gcs_uri: str) -> PDFContent:
    """Loads the text content of a PDF document from Google Cloud Storage into a structured PDFContent object."""
    logger.info(f"Loading PDF from GCS: {gcs_uri}")
    try:
        logger.debug(f"Parsing GCS URI: {gcs_uri}")
        parsed_uri = urlparse(gcs_uri)
        bucket_name = parsed_uri.netloc
        blob_name = parsed_uri.path.lstrip("/")
        logger.debug(f"Bucket name: {bucket_name}, Blob name: {blob_name}")

        logger.debug("Initializing GCSFileLoader.")
        loader = GCSFileLoader(bucket=bucket_name, blob=blob_name, project_name="bhagavangenai-444212")
        logger.debug("GCSFileLoader initialized.")

        logger.debug("Loading pages from GCS.")
        pages = loader.load()
        logger.debug(f"Number of pages loaded: {len(pages)}")

        logger.debug("Extracting text content from pages.")
        text = "\n".join([page.page_content for page in pages])
        logger.debug(f"Text content extracted. Length: {len(text)}")

        logger.debug("PDF loaded successfully from GCS.")
        return PDFContent(text=text)
    except Exception as e:
        logger.exception(f"Error loading PDF from GCS: {e}")
        raise PDFLoadError(f"Error loading PDF from GCS: {e}")

# --- MODIFIED FUNCTION: Load PDF from Local Path using UnstructuredPDFLoader ---
def load_pdf_content_local(local_path: str) -> PDFContent:
    """Loads the text content of a PDF document from a local file path into a structured PDFContent object using UnstructuredPDFLoader."""
    logger.info(f"Loading PDF from local path: {local_path}")
    try:
        logger.debug("Initializing UnstructuredPDFLoader.")
        loader = UnstructuredPDFLoader(local_path)  # Use UnstructuredPDFLoader
        logger.debug("UnstructuredPDFLoader initialized.")

        logger.debug("Loading document with UnstructuredPDFLoader")
        document = loader.load()  # UnstructuredPDFLoader returns a list of Document objects
        logger.debug(f"Number of documents loaded: {len(document)}")

        logger.debug("Extracting text content from documents.")
        text_content = ""
        for doc in document:  # Iterate through Document objects
            text_content += doc.page_content + "\n"  # Concatenate page_content from each doc
        logger.debug(f"Text content extracted. Length: {len(text_content)}")

        return PDFContent(text=text_content)
    except Exception as e:
        logger.exception(f"Error loading PDF from local path: {e}")
        raise PDFLoadError(f"Error loading PDF from local path using UnstructuredPDFLoader: {e}")

def clean_llm_output(llm_output: str) -> str:
    """
    Cleans the LLM output by removing any leading/trailing whitespace and ```json blocks.
    """
    logger.debug("Cleaning LLM output.")
    logger.debug(f"Original LLM output: {llm_output}")

    llm_output = llm_output.strip()
    logger.debug(f"Stripped LLM output: {llm_output}")

    # Remove ```json and ``` if present
    if llm_output.startswith("```json"):
        llm_output = llm_output[len("```json"):].strip()
        logger.debug("Removed ```json prefix.")
    if llm_output.endswith("```"):
        llm_output = llm_output[:-len("```")].strip()
        logger.debug("Removed ``` suffix.")

    logger.debug("LLM output cleaned.")
    return llm_output

def save_assessment_to_json(assessment: Assessment, filename: str):
    """Saves the assessment data to a JSON file."""
    logger.info(f"Saving assessment to JSON: {filename}")
    try:
        logger.debug(f"Serializing assessment to JSON string.")
        json_string = assessment.json(indent=2) # Use pydantic's .json() method
        logger.debug("Assessment serialized to JSON string.")
        with open(filename, "w") as f:
            logger.debug("Writing assessment to file.")
            f.write(json_string)
        logger.debug("Assessment saved successfully.")
        print(f"Assessment saved to {filename}")  # Keep this for user feedback
    except Exception as e:
        logger.exception(f"Error saving assessment to JSON: {e}")
        print(f"Error saving assessment to JSON: {e}")  # Keep this for user feedback

def save_assessment_to_file(assessment_text: str, subject: str = "mathematics") -> None:
    logger.debug(f"In save_assessment_to_file...")
    logger.debug(f"assessment_text len :{len(assessment_text)}")
    if assessment_text:
        try:
            assessments_dir = "assessments"
            if not os.path.exists(assessments_dir):
                os.makedirs(assessments_dir)
                logger.debug(f"Created directory: {assessments_dir}")
            logger.info("Assessments dir created") 

        except Exception as e:
            logger.exception(f"Error creating directory: {e}")
            return None

    now = datetime.datetime.now()
    date_time_string = now.strftime("%Y%m%d_%H%M%S")
    subject = "mathematics"
    filename = os.path.join(assessments_dir, f"{date_time_string}_{subject}_assessment.txt")
    logger.debug(f"Filename generated: {filename}")

    try:
        with open(filename, "w") as f:
            logger.debug("Writing content to file.")
            f.write(assessment_text)
        logger.info(f"Assessment saved to: {filename}")
    except Exception as e:
        logger.exception(f"Error writing to file: {e}")
            