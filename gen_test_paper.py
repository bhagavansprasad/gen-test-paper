from summary import summarize_test_paper
import json
from test_paper import generate_assessment_from_gcs


test_paper_summary = {
    "stp_analysis": {
        "all_compulsory": True,
        "internal_choice_count": {
            "B": 2,
            "C": 2,
            "D": 2,
            "E": 3
        },
        "questions_distrubution_type": {
            "descriptive": 18,
            "mcqs": 20
        },
        "section_breakdown": {
            "A": {
                "internal_choice": False,
                "mark_per_question": 1,
                "num_questions": 20,
                "question_type": "MCQs and Assertion-Reason",
                "total_marks": 20
            },
            "B": {
                "internal_choice": True,
                "mark_per_question": 2,
                "num_questions": 5,
                "question_type": "VSA",
                "total_marks": 10
            },
            "C": {
                "internal_choice": True,
                "mark_per_question": 3,
                "num_questions": 6,
                "question_type": "SA",
                "total_marks": 18
            },
            "D": {
                "internal_choice": True,
                "mark_per_question": 5,
                "num_questions": 4,
                "question_type": "LA",
                "total_marks": 20
            },
            "E": {
                "internal_choice": True,
                "mark_per_question": 4,
                "num_questions": 3,
                "question_type": "Case-Study Based",
                "total_marks": 12
            }
        },
        "sections": [
            "A",
            "B",
            "C",
            "D",
            "E"
        ],
        "total_questions": 38
    }
}

def main():
    gcs_pdf_uri = "gs://bhagavan-pub-bucket/aignite-resources/jemh1a1.pdf"  # Replace with your GCS URI
    test_paper_path = "gs://bhagavan-pub-bucket/aignite-resources/sample-test-paper1.pdf" # GCS path for sample test paper
    # test_paper_path = "testpapers/sample-test-paper3.pdf" 

    # Summarize the sample test paper
    # test_paper_summary = summarize_test_paper(test_paper_path)
    if test_paper_summary:
        print("\n--- Test Paper Summary ---")
        print(json.dumps(test_paper_summary, indent=4, sort_keys=True))
    else:
        print("\n--- Failed to summarize test paper ---")

    assessment = generate_assessment_from_gcs(gcs_pdf_uri, test_paper_summary)

    if assessment:
        print("\n--- Generated Test Paper ---")
        print(assessment)
    else:
        print("Failed to generate assessment.")


if __name__ == "__main__":
    main()
