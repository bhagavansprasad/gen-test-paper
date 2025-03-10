from models import PDFContent, MCQ, Summary, Assessment
from urllib.parse import urlparse
from langchain_google_community import GCSFileLoader
from pdbwhereami import whereami
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer #Add new
from reportlab.lib.styles import getSampleStyleSheet # New Import
from reportlab.lib.units import inch #added new
from reportlab.lib.styles import ParagraphStyle #added new
from reportlab.lib import colors #new dep
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from langchain_community.document_loaders import PyPDFLoader # Import PyPDFLoader
from pdbwhereami import whereami
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer #Add new
from reportlab.lib.styles import getSampleStyleSheet # New Import
from reportlab.lib.units import inch #added new
from reportlab.lib.styles import ParagraphStyle #added new
from reportlab.lib import colors #new dep
from langchain_community.document_loaders import UnstructuredPDFLoader
from pdbwhereami import whereami

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

# --- MODIFIED FUNCTION: Load PDF from Local Path using UnstructuredPDFLoader ---
def load_pdf_content_local(local_path: str) -> PDFContent:
    """Loads the text content of a PDF document from a local file path into a structured PDFContent object using UnstructuredPDFLoader."""
    whereami(local_path) #debug path
    try:
        # loader = PyPDFLoader(local_path) # Removed PyPDFLoader
        loader = UnstructuredPDFLoader(local_path) # Use UnstructuredPDFLoader
        whereami("UnstructuredPDFLoader initialized") # Debugging point - Changed message
        document = loader.load() # UnstructuredPDFLoader returns a list of Document objects
        whereami(f"Number of documents loaded: {len(document)}") # Debugging - documents, not pages

        # --- Extract text from Document objects ---
        text_content = ""
        for doc in document: # Iterate through Document objects
            text_content += doc.page_content + "\n" # Concatenate page_content from each doc
        
        whereami(f"PDF Content length: {len(text_content)}") # Debugging point - length of extracted text
        # whereami(f"PDF Content: {text_content[:200]}...") # Print the first 200 characters
        return PDFContent(text=text_content) # Use text_content
    except Exception as e:
        print(f"Error loading PDF from local path using UnstructuredPDFLoader: {e}") # Changed error message
        whereami(f"Exception details: {e}") # Print exception details
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
