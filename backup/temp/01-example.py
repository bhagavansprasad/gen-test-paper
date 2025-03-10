from typing import List, Dict
from langchain_core.runnables import chain
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_vertexai import VertexAI
from pydantic import BaseModel, Field

# 1. Define the Structured Output (Pydantic Model)
class Person(BaseModel):
    """Represents a person with a name, age, and occupation."""
    name: str = Field(description="The person's full name.")
    age: int = Field(description="The person's age in years.")
    occupation: str = Field(description="The person's current job or profession.")
    is_student: bool = Field(description="Whether the person is a student or not")

# 2. Define the Input Data (Dictionary -> String for Simplicity)

# 3. Define the Language Model (Gemini in this case)
try:
    llm = VertexAI(model_name="gemini-pro", temperature=0.5)
    print("Using gemini-pro")
except Exception as e:
    print(f"Failed to initialize gemini-pro: {e}.  Falling back to gemini-pro.")
    llm = VertexAI(model_name="gemini-pro", temperature=0.5)  # Fallback
    print("Using gemini-pro")


# 4. Define the Output Parser
parser = PydanticOutputParser(pydantic_object=Person)

# 5.  Define a Simple Transformation Function (Instead of a Complex Prompt)
def transform_data(data: Dict) -> str:
    """Transforms a dictionary of attributes into a simple descriptive string,
    explicitly requesting a JSON output."""
    return f"Create a JSON object representing a person with the following attributes: {data}. The JSON object should have the keys 'name', 'age', 'occupation', and 'is_student'. Ensure the 'age' is an integer and 'is_student' is a boolean.  Return *only* the JSON, nothing else."

# 6. Construct the Langraph Chain
graph_chain = (
    transform_data # Transform the input data
    | llm
    | parser  # Parse into the Pydantic Model
)

# 7. Example Usage

input_data = {
    "name": "Alice Smith",
    "age": 22,
    "occupation": "Student",
    "is_student": True
}

try:
    person: Person = graph_chain.invoke(input_data)
    print("Generated Person Object:")
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Occupation: {person.occupation}")
    print(f"Is Student: {person.is_student}")

except Exception as e:
    print(f"An error occurred: {e}")