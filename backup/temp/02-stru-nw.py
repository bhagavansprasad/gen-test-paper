from typing import List, Optional
from pydantic import BaseModel, Field
import os
from langchain_google_vertexai import VertexAI
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph

# Define structured input schema
class DocumentInput(BaseModel):
    title: str = Field(description="Title of the document")
    content: str = Field(description="Full text content of the document")
    author: str = Field(description="Author of the document")
    category: str = Field(description="Category of the document")

# Define structured output schema
class SummaryOutput(BaseModel):
    title: str = Field(description="Title of the document")
    summary: str = Field(description="Concise summary of the document")
    key_points: List[str] = Field(description="List of important points extracted")

# Define the overall state schema
class WorkflowState(BaseModel):
    document: DocumentInput
    summary_result: Optional[SummaryOutput] = None  # Initially None

# Initialize the LLM (Gemini via Vertex AI)
llm = VertexAI(model_name="gemini-pro", temperature=0.3)

# Output Parser
output_parser = PydanticOutputParser(pydantic_object=SummaryOutput)

from langchain_core.prompts import PromptTemplate

def summarize_text(state: WorkflowState) -> WorkflowState:
    """Generates a structured summary with key points."""
    doc = state.document  # Extract input

    # Convert structured input into a formatted string
    prompt_template = PromptTemplate.from_template(
        "Title: {title}\nContent: {content}\nAuthor: {author}\nCategory: {category}\n\nSummarize this document."
    )

    prompt_str = prompt_template.format(
        title=doc.title,
        content=doc.content,
        author=doc.author,
        category=doc.category,
    )

    # Invoke LLM with string prompt
    structured_output = output_parser.parse(llm.invoke(prompt_str))

    return WorkflowState(document=doc, summary_result=structured_output)

# Define LangGraph Workflow
workflow = StateGraph(WorkflowState)  # ✅ Use Pydantic Model

# Add nodes
workflow.add_node("summarize_text", summarize_text)

# Set the **entry point** and **remove explicit END node**
workflow.set_entry_point("summarize_text")  # ✅ No need for END
# ❌ Remove `workflow.add_edge("summarize_text", "END")`

# Compile the graph
app = workflow.compile()

if __name__ == "__main__":
    # Example structured input
    doc_input = DocumentInput(
        title="AI in Healthcare",
        content="Artificial Intelligence is transforming healthcare by enabling predictive analytics, personalized treatments, and efficient diagnostics.",
        author="Dr. Rajan",
        category="Technology"
    )

    # Run the Langraph workflow
    result = app.invoke(WorkflowState(document=doc_input))

    # Print structured output
    print(result.summary_result)
