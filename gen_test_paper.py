from summary import summarize_test_paper
import json
from test_paper import generate_assessment_from_gcs

def main():
    gcs_pdf_uri = "gs://bhagavan-pub-bucket/aignite-resources/jemh1a1.pdf"  # Replace with your GCS URI
    test_paper_path = "gs://bhagavan-pub-bucket/aignite-resources/sample-test-paper1.pdf" # GCS path for sample test paper

    # Summarize the sample test paper
    test_paper_summary = summarize_test_paper(test_paper_path)
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
