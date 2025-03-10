import os
from typing import List, Dict, Optional
import re
import random

from langchain_google_vertexai import VertexAI
from langchain_core.runnables import chain
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field

# --- Data Models ---
class PDFContent(BaseModel):
    """Represents the content of a PDF document."""
    text: str = Field(description="The textual content extracted from the PDF.")


class Chapter(BaseModel):
    """Represents a chapter or section within a document."""
    title: str = Field(description="The title of the chapter.")
    content: str = Field(description="The content of the chapter.")  # Add chapter content
    number: int = Field(description="The Chapter Number")


class MCQ(BaseModel):
    """Represents a Multiple Choice Question."""
    question: str = Field(description="The question text.")
    a: str = Field(description="Option A")  # Changed to 'a'
    b: str = Field(description="Option B")  # Changed to 'b'
    c: str = Field(description="Option C")  # Changed to 'c'
    d: str = Field(description="Option D")  # Changed to 'd'
    complexity: str = Field(description="Complexity level (easy, medium, tough).")
    answer: str = Field(description="The correct answer (A, B, C, or D).")
    from_chapter: str = Field(description="The chapter the question is from.")
    page_number: int = Field(description="The page number the question is from.")  # Estimate
    marks: int = Field(description="Marks allocated for the question.")


# --- Helper Functions ---
def load_pdf_content(pdf_path: str) -> PDFContent:
    """Loads the text content of a PDF document into a structured PDFContent object."""
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
        return PDFContent(text=text)
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return PDFContent(text="")


def extract_chapters(pdf_content: PDFContent) -> List[Chapter]:
    """Extracts chapters from the PDF content using regex."""
    chapter_pattern = r"Chapter\s*(\d+):\s*(.+?)(?=Chapter\s*\d+|$)"
    matches = re.findall(chapter_pattern, pdf_content.text, re.DOTALL)
    chapters = []
    for match in matches:
        chapter_number = int(match[0].strip())
        chapter_title = match[1].split('\n', 1)[0].strip()
        chapter_content = '\n'.join(match[1].split('\n')[1:]).strip()
        chapters.append(Chapter(title=chapter_title, content=chapter_content, number=chapter_number))
    return chapters


def generate_mcq(chapter: Chapter) -> MCQ:
    """Generates an MCQ for a given chapter using the LLM."""
    llm = VertexAI(model_name="gemini-pro", temperature=0.7) # Increase temperature for variety

    prompt_data = {
        "chapter_title": chapter.title,
        "chapter_content": chapter.content,
        "instruction": """Generate a single multiple-choice question (MCQ) based on the chapter. The MCQ should have the following format:
            Question: [Question Text]
            A: [Option A Text]
            B: [Option B Text]
            C: [Option C Text]
            D: [Option D Text]
            Complexity: (easy, medium, tough)
            Answer: (A, B, C, or D)
            FromChapter: {chapter_title}
            PageNumber: (Estimate - Provide a realistic page number for this content)
            Marks: (1 or 2)

            The question should be relevant to the chapter content and have only one correct answer.
            Ensure the question, options, and other fields are complete and well-formatted.
            Only the above information should be returned with the exact key names above.
            """
    }

    # Construct the prompt from the structured data
    prompt = f"""
    Based on: Chapter Title: {prompt_data["chapter_title"]}, Chapter Content: {prompt_data["chapter_content"]},
    follow these instructions: {prompt_data["instruction"]}
    """

    llm_response = llm.invoke(prompt)
    # Parse the LLM response (This is a simple parsing. Improve as needed)
    try:
        question_match = re.search(r"Question:\s*(.+)", llm_response)
        a_match = re.search(r"A:\s*(.+)", llm_response)          # Changed Option A -> A
        b_match = re.search(r"B:\s*(.+)", llm_response)          # Changed Option B -> B
        c_match = re.search(r"C:\s*(.+)", llm_response)          # Changed Option C -> C
        d_match = re.search(r"D:\s*(.+)", llm_response)          # Changed Option D -> D
        complexity_match = re.search(r"Complexity:\s*(easy|medium|tough)", llm_response)
        answer_match = re.search(r"Answer:\s*([A-D])", llm_response)
        page_number_match = re.search(r"PageNumber:\s*(\d+)", llm_response)
        marks_match = re.search(r"Marks:\s*(\d+)", llm_response)

        if not all([question_match, a_match, b_match, c_match, d_match,
                    complexity_match, answer_match, page_number_match, marks_match]):
            print(f"Incomplete MCQ information from LLM:\n{llm_response}")
            return None

        question = question_match.group(1).strip()
        a = a_match.group(1).strip() # Changed option_a -> a
        b = b_match.group(1).strip() # Changed option_b -> b
        c = c_match.group(1).strip() # Changed option_c -> c
        d = d_match.group(1).strip() # Changed option_d -> d
        complexity = complexity_match.group(1).strip()
        answer = answer_match.group(1).strip()
        page_number = int(page_number_match.group(1).strip())
        marks = int(marks_match.group(1).strip())

        return MCQ(question=question, a=a, b=b, c=c,
                   d=d, complexity=complexity, answer=answer,
                   from_chapter=chapter.title, page_number=page_number, marks=marks)

    except Exception as e:
        print(f"Error parsing MCQ: {e}\nLLM Response: {llm_response}")
        return None
    

# --- Main Function ---
def generate_mcqs(pdf_path: str, num_mcqs: int = 30) -> List[MCQ]:
    """Generates a list of MCQs from a PDF document."""
    try:
        pdf_content = load_pdf_content(pdf_path)
        if not pdf_content.text:
            return []

        chapters = extract_chapters(pdf_content)
        if not chapters:
            print("No chapters found in the PDF.")
            return []

        mcqs = []
        chapter_index = 0
        while len(mcqs) < num_mcqs and chapter_index < len(chapters):
            chapter = chapters[chapter_index]
            mcq = generate_mcq(chapter)
            if mcq:
                mcqs.append(mcq)
            chapter_index += 1
            if chapter_index == len(chapters): #reset and regenerate
                chapter_index = 0


        if len(mcqs) < num_mcqs:
            print(f"Could only generate {len(mcqs)} MCQs from the PDF.")

        return mcqs

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# --- Example Usage ---
if __name__ == "__main__":
    pdf_file_path = "lesson.pdf" 

    mcqs = generate_mcqs(pdf_file_path, num_mcqs=30)

    if mcqs:
        print("Generated MCQs:")
        for i, mcq in enumerate(mcqs):
            print(f"MCQ {i+1}:")
            print(f"Question: {mcq.question}")
            print(f"A: {mcq.a}") # Changed option_a -> a
            print(f"B: {mcq.b}") # Changed option_b -> b
            print(f"C: {mcq.c}") # Changed option_c -> c
            print(f"D: {mcq.d}") # Changed option_d -> d
            print(f"Complexity: {mcq.complexity}")
            print(f"Answer: {mcq.answer}")
            print(f"From Chapter: {mcq.from_chapter}")
            print(f"Page Number: {mcq.page_number}")
            print(f"Marks: {mcq.marks}")
            print("-" * 20)
    else:
        print("Failed to generate MCQs.")
