from typing import List, Dict, Optional, Literal
from langchain_core.prompts import PromptTemplate #new import
from common import load_pdf_content_from_gcs, summarize_test_paper
from langchain_google_vertexai import VertexAI
from common import clean_llm_output

# --- NEW FUNCTION: Summarize Test Paper ---
def summarize_test_paper(test_paper_path: str) -> Optional[str]:
    """Summarizes a test paper using an LLM."""
    try:
        # 1. Load the Prompt
        prompt_path = "prompts/01-summarize-testpaper.prompt"
        with open(prompt_path, "r") as f:
            template = f.read()

        prompt_template = PromptTemplate.from_template(template)

        # 2. Load the Test Paper (PDF)
        pdf_content = load_pdf_content_from_gcs(test_paper_path) # Assuming GCS path, adjust if needed
        if not pdf_content.text:
            print(f"Could not load test paper content from: {test_paper_path}")
            return None

        # 3. Create the LLM
        llm = VertexAI(model_name="gemini-1.5-pro-002", temperature=0.2) # Adjust model and temp

        # 4. Format the Prompt
        formatted_prompt = prompt_template.format(test_paper_text=pdf_content.text)

        # 5. Invoke the LLM
        llm_response = llm.invoke(formatted_prompt)

        # 6. Clean and Return the Output
        cleaned_output = clean_llm_output(llm_response)
        print(f"Test paper summary:{cleaned_output}")
        return cleaned_output

    except Exception as e:
        print(f"Error summarizing test paper: {e}")
        return None
