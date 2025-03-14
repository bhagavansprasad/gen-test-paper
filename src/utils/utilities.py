import os
import datetime
import logging  # Add this import
from src.models import PDFContent, Assessment
from urllib.parse import urlparse
from langchain_google_community import GCSFileLoader
import json
from typing import Dict
from langchain_community.document_loaders import UnstructuredPDFLoader
import graphviz
import os
from langgraph.graph import StateGraph, START


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
    logger.debug(f"Original LLM output len: {len(llm_output)}")

    llm_output = llm_output.strip()

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

def save_summary_to_file(summary: Dict, subject: str = "mathematics") -> None:
    """Saves the test paper summary to a JSON file with a timestamped name in the 'summaries' directory."""
    logger.info("Saving test paper summary to file...")
    try:
        if summary:
            summaries_dir = "assessments"
            if not os.path.exists(summaries_dir):
                os.makedirs(summaries_dir)
                logger.debug(f"Created directory: {summaries_dir}")
            logger.info("Summaries dir created") 
        else:
            logger.warning("No test paper summary found")
            return 

    except Exception as e:
        logger.exception(f"Error creating directory: {e}")
        return None

    now = datetime.datetime.now()
    date_time_string = now.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(summaries_dir, f"{date_time_string}_{subject}_summary.json")
    logger.debug(f"Filename generated: {filename}")

    try:
        with open(filename, "w") as f:
            logger.debug("Writing content to file.")
            json.dump(summary, f, indent=4) 
        logger.info(f"Test paper summary saved to: {filename}")
    except Exception as e:
        logger.exception(f"Error writing to file: {e}")

def save_assessment_to_file(assessment_text: str, subject: str = "mathematics") -> None:
    """Saves the assessment text to a file with a timestamped name in the 'assessments' directory."""
    logger.info("Saving assessment to file...")
    try:
        if assessment_text:
            assessments_dir = "assessments"
            if not os.path.exists(assessments_dir):
                os.makedirs(assessments_dir)
                logger.debug(f"Created directory: {assessments_dir}")
            logger.info("Assessments dir created") 
        else:
            logger.warning("No assessment found")
            return 

    except Exception as e:
        logger.exception(f"Error creating directory: {e}")
        return None

    now = datetime.datetime.now()
    date_time_string = now.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(assessments_dir, f"{date_time_string}_{subject}_assessment.json")
    logger.debug(f"Filename generated: {filename}")

    assessment_text = assessment_text.strip('`')
    assessment_text = assessment_text.strip('json')

    try:
        assessment_data = json.loads(assessment_text)

        with open(filename, "w") as f:
            logger.debug("Writing content to file.")
            json.dump(assessment_data, f, indent=4)  # Dump the parsed object
        logger.info(f"Assessment saved to: {filename}")
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {e}.  Invalid JSON string: {assessment_text}")
        logger.exception(e)
    except Exception as e:
        logger.exception(f"Error writing to file: {e}")


def get_entry_point_node_name(graph: StateGraph) -> str | None:
  try:
    # Iterate through the edges to find the edge starting from the START node
    for start_node, node_name in graph.edges:
        if start_node == START:
            return node_name  # Found the entry point
    return None  # No entry point found
  except Exception as e:
    print(f"Error retrieving entry point: {e}")
    return None

def visualize_graph(graph, filename="graph"):
    """
    Visualizes a LangGraph graph using graphviz and saves it as an image.

    Args:
        graph_data: A dictionary containing the graph definition (nodes, edges, entry_point).
        filename: The name of the image file to save (e.g., "graph.png", "graph.pdf").
    """
    
    filename = filename.split(".")[0]
    print(f"Visualizing graph and saving to {filename}")

    entry_point = get_entry_point_node_name(graph)
    
    graph_data = {
        "nodes": list(graph.nodes.keys()),
        "edges": graph.edges,
        "entry_point": entry_point
    }
    
    try:
        dot = graphviz.Digraph(comment='LangGraph Graph')

        # Add nodes
        for node_name in graph_data["nodes"]:
            dot.node(node_name, node_name)  # Use node name as label

        # Add edges
        for start_node, end_node in graph_data["edges"]:
            dot.edge(start_node, end_node)

        # Set graph attributes (optional)
        dot.attr(rankdir='LR')  # Left-to-right layout

        # Save the graph to a file
        dot.render(filename, view=False, format='png', cleanup=True)
        print(f"Graph visualized and saved to {filename}")

    except Exception as e:
        print(f"Error visualizing graph: {e}")
            