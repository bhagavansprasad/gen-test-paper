from typing import List, Dict, Optional, Literal
from langchain_core.prompts import PromptTemplate  # new import
from common import load_pdf_content_from_gcs, load_pdf_content_local
from langchain_google_vertexai import VertexAI
from common import clean_llm_output
from pdbwhereami import whereami
import json  # Import for JSON handling


def summarize_test_paper(test_paper_path: str) -> Optional[Dict]:  # Changed return type to Optional[Dict]
    """Summarizes a test paper using an LLM and returns JSON."""
    try:
        # 1. Load the Prompt
        prompt_path = "prompts/01-summarize-testpaper.prompt"
        with open(prompt_path, "r") as f:
            template = f.read()

        # --------------------- PROMPT TEMPLATE DEFINITION (CHANGED) ---------------------
        prompt_template = PromptTemplate(
            input_variables=["test_paper_text"], # Explicitly define input variable
            template=template,
        )
        # -------------------------------------------------------------------------------
        # whereami(prompt_template) # Keep this for debugging
        whereami()

        # 2. Load the Test Paper (PDF)
        if test_paper_path.startswith("gs://"):
            pdf_content = load_pdf_content_from_gcs(test_paper_path)
        else: # Assume it's a local path
            pdf_content = load_pdf_content_local(test_paper_path)

        if not pdf_content.text:
            print(f"Could not load test paper content from: {test_paper_path}")
            return None

        whereami(f"pdf_content.text :{pdf_content.text[:100]}...") # Print the first 100 chars

        try:
            formatted_prompt = prompt_template.format(test_paper_text=pdf_content.text) 

            whereami()
        except Exception as e:
            print(f"Error formatting prompt: {e}")
            return None
        
        whereami()
        # 3. Create the LLM
        llm = VertexAI(model_name="gemini-1.5-pro-002", temperature=0.2)  # Adjust model and temp
        
        # 5. Invoke the LLM
        llm_response = llm.invoke(formatted_prompt)

        whereami(f"LLM Response: {llm_response}")

        # 6. Clean and Parse JSON Output
        cleaned_output = clean_llm_output(llm_response)
        # print(f"Test paper summary:{cleaned_output}")

        try:
            summary_json = json.loads(cleaned_output)  # Parse JSON
            return summary_json
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            print(f"LLM Output (for debugging): {cleaned_output}")  # Print the raw LLM output for inspection
            return None  # Return None if parsing fails

    except Exception as e:
        print(f"Error summarizing test paper: {e}")
        return None