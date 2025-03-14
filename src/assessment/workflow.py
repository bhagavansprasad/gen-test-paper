from typing import Dict, Any, Optional
from langchain_core.runnables import chain
from langgraph.graph import StateGraph, END
from src.models import PDFContent, Assessment  # Import necessary models
from src.assessment.analyzer import summarize_test_paper  # Import node functions
from src.assessment.generator import generate_assessment  # Import node functions
from src.utils.utilities import load_pdf_content_from_gcs, PDFLoadError  # Import common functions
from dataclasses import dataclass
from src.utils.logging_config import configure_logging
from src.utils.utilities import save_assessment_to_file, save_summary_to_file
import logging
import os
from src.utils.utilities import visualize_graph

configure_logging()
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# 1. Define the State
# ----------------------------------------------------------------------------

@dataclass
class GraphState:
    gcs_pdf_uri: str
    test_paper_path: str
    pdf_content: Optional[PDFContent] = None
    test_paper_summary: Optional[Dict] = None
    assessment: Optional[Assessment] = None
    error: Optional[str] = None  # Add an error field

    def __post_init__(self):
        logger.debug(f"GraphState initialized: gcs_pdf_uri={self.gcs_pdf_uri}, test_paper_path={self.test_paper_path}")



# ----------------------------------------------------------------------------
# 2. Define the Nodes
# ----------------------------------------------------------------------------

def summarize_node(state: GraphState):
    """Summarizes the test paper."""
    logger.debug("Entering summarize_node")
    logger.debug(f"Current state: {state}")
    try:
        logger.info(f"Summarizing test paper: {state.test_paper_path}")
        state.test_paper_summary = summarize_test_paper(state.test_paper_path)

        if state.test_paper_summary:
            logger.info("Test paper summarized successfully.")
            logger.debug(f"Test paper summary: {state.test_paper_summary}")
            logger.info("\n--- Test Paper Summary ---")
            logger.info(state.test_paper_summary)
        
            try:
                save_summary_to_file(state.test_paper_summary)
                logger.info("Test paper summary saved to file.")
            except Exception as e:
                logger.exception(f"Error saving test paper summary to file: {e}")
                
        else:
            logger.warning("Failed to summarize test paper.")
            state.error = "Failed to summarize test paper"
            logger.debug(f"State after summarize_node: {state}")
            return state # Exit the node here if summarization failed

        logger.debug(f"State after summarize_node: {state}")
        return state #Exit the node here
    except Exception as e:
        logger.exception(f"Error during summarization: {e}")
        state.error = f"Error during summarization: {e}"
        logger.debug(f"State after exception in summarize_node: {state}")
        return state #Exit the node here
    finally:
        logger.debug("Exiting summarize_node") #Exit here

def load_chapter_content_node(state: GraphState):
    """Loads chapter content from GCS."""
    logger.debug("Entering load_chapter_content_node")
    logger.debug(f"Current state: {state}")
    try:
        logger.info(f"Loading chapter content from GCS: {state.gcs_pdf_uri}")
        state.pdf_content = load_pdf_content_from_gcs(state.gcs_pdf_uri)

        if state.pdf_content and state.pdf_content.text:
            logger.info("Chapter content loaded successfully.")
            logger.debug(f"Chapter content (first 100 chars): {state.pdf_content.text[:100]}...")
        else:
            logger.warning("Failed to load chapter content.")
            state.error = "Failed to load chapter content"
            logger.debug(f"State after load_chapter_content_node failure: {state}")
            return state # Exit the node here if content loading failed.

        logger.debug(f"State after load_chapter_content_node: {state}")
        return state #Exit the node here
    except PDFLoadError as e:
        logger.exception(f"PDF loading error: {e}")
        state.error = f"PDF loading error: {e}"
        logger.debug(f"State after PDFLoadError in load_chapter: {state}")
        return state #Exit the node here
    except Exception as e:
        logger.exception(f"Unexpected error loading content: {e}")
        state.error = f"Unexpected error loading content: {e}"
        logger.debug(f"State after exception in load_chapter: {state}")
        return state #Exit the node here
    finally:
        logger.debug("Exiting load_chapter_content_node") #Exit here


def generate_assessment_node(state: GraphState):
    """Generates the assessment."""
    logger.debug("Entering generate_assessment_node")
    logger.debug(f"Current state: {state}")
    try:
        logger.info("Generating assessment...")
        if not (state.pdf_content and state.test_paper_summary):
            logger.warning("Missing PDF content or test paper summary. Cannot generate assessment.")
            state.error = "Missing PDF content or test paper summary. Cannot generate assessment."
            logger.debug(f"State after generate_assessment_node failure (missing data): {state}")
            return state #Exit the node here

        logger.debug("Calling generate_assessment function.")
        state.assessment = generate_assessment(state.pdf_content, state.test_paper_summary, [state.gcs_pdf_uri])

        if state.assessment:
            logger.info("Assessment generated successfully.")
            logger.debug("\n--- Generated Test Paper ---")
            logger.debug(f"Assessment len: {len(state.assessment)}")
        else:
            logger.warning("Failed to generate assessment.")
            state.error = "Failed to generate assessment"
            logger.debug(f"State after generate_assessment_node failure: {state}")
            return state #Exit the node here
        logger.debug(f"State after generate_assessment_node: {state}")
        return state #Exit the node here

    except Exception as e:
        logger.exception(f"Error during assessment generation: {e}")
        state.error = f"Error during assessment generation: {e}"
        logger.debug(f"State after exception in generate_assessment_node: {state}")
        return state #Exit the node here
    finally:
        logger.debug("Exiting generate_assessment_node") #Exit here


def finished_node(state: GraphState):
    """Final node; prints results and handles any cleanup."""
    logger.debug("Entering finished_node")
    logger.debug(f"Current state: {state}")
    logger.info("Assessment generation completed.")
    if state.error:
        logger.error(f"An error occurred: {state.error}")
    else:
        logger.info("Assessment generated successfully.")

    # Print the assessment
    if state.assessment:
        logger.info
        ("\n--- Generated Test Paper ---")
        # logger.info(state.assessment)

    # Save the assessment to a text file
    if state.assessment:
        try:
            # Call the utility function to save the file
            save_assessment_to_file(state.assessment)
        except Exception as e:
            logger.exception(f"Error saving assessment to file: {e}")

    logger.debug(f"State after finished_node: {state}")
    logger.debug("Exiting finished_node")
    
    return state #Exit here

# ----------------------------------------------------------------------------
# 3. Define the Graph
# ----------------------------------------------------------------------------

def create_graph():
    logger.debug("Creating LangGraph graph.")
    builder = StateGraph(GraphState)

    builder.add_node("summarize", summarize_node)
    builder.add_node("load_chapter", load_chapter_content_node)
    builder.add_node("generate_assessment", generate_assessment_node)
    builder.add_node("finished", finished_node)
    logger.debug("Nodes added to graph.")

    # Define edges using add_conditional_edges correctly
    builder.add_conditional_edges(
        source="summarize",
        path=lambda state: "load_chapter" if state.test_paper_summary else "finished",
    )
    logger.debug("Conditional edge added: summarize -> load_chapter or finished")

    builder.add_conditional_edges(
        source="load_chapter",
        path=lambda state: "generate_assessment" if state.pdf_content else "finished",
    )
    logger.debug("Conditional edge added: load_chapter -> generate_assessment or finished")

    builder.add_conditional_edges(
        source="generate_assessment",
        path=lambda state: "finished",  # Always go to finished.  No need for a condition.
    )
    logger.debug("Edge added: generate_assessment -> finished")

    builder.set_entry_point("summarize")
    logger.debug("Entry point set to 'summarize'")
   
    return builder

# ----------------------------------------------------------------------------
# 4. Main Execution
# ----------------------------------------------------------------------------

def main():
    logger.debug("Entering main function.")
    gcs_pdf_uri = "gs://bhagavan-pub-bucket/aignite-resources/jemh1a1.pdf"  # Replace with your GCS URI
    test_paper_path = "gs://bhagavan-pub-bucket/aignite-resources/sample-test-paper1.pdf"  # GCS path for sample test paper
    logger.debug(f"GCS PDF URI: {gcs_pdf_uri}, Test paper path: {test_paper_path}")

    # Initialize the graph state
    logger.debug("Initializing GraphState.")
    initial_state = GraphState(gcs_pdf_uri=gcs_pdf_uri, test_paper_path=test_paper_path)
    logger.debug(f"Initial state: {initial_state}")

    # Create and execute the graph
    logger.debug("Creating graph.")
    builder = create_graph()

    visualize_graph(builder, filename=os.path.basename(__file__))

    print(f"builder :{dir(builder)}")
    graph = builder.compile()
    logger.info("LangGraph graph created and compiled.")
    
    print(f"graph :{dir(graph)}")
    logger.debug("Graph created. Invoking graph with initial state.")

    results = graph.invoke(initial_state)

    logger.debug("Final Results...")
    logger.info(f"Graph execution completed")
    logger.debug("Exiting main function.")


if __name__ == "__main__":
    main()
