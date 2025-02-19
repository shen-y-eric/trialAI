#!/usr/bin/env python3

import os
import argparse
import requests
import json
import openai
import io
import pandas as pd

def extract_key_terms(discharge_summary: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Uses the OpenAI ChatCompletion API to extract key search terms 
    from a medical discharge summary. Returns a comma-separated list.
    """
    # We create a list of messages according to the ChatCompletion format.
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority.  Please output only a JSON dict formatted as Dict{{'summary': Str(summary), 'conditions': List[Str(condition)]}}."
            )
        },
        {
            "role": "user",
            "content": f"Here is the patient description:\n\n{discharge_summary}"
        }
    ]

    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        n=1
    )

    key_terms = response.choices[0].message.content.strip()
    return key_terms


def build_ctgov_query(
    base_url: str,
    parsed_terms: dict,
    page_size: int = 10,
    max_results: int = None,
    output_format: str = "csv"
) -> str:
    """
    Build the query string for the new clinicaltrials.gov v2/studies endpoint,
    enforcing certain constant filters:
      - overallStatus = NOT_YET_RECRUITING,ENROLLING_BY_INVITATION,RECRUITING
      - sort = @relevance
      - trials in the U.S. (query.locn=USA)

    :param base_url: Base URL for the clinicaltrials.gov API (e.g. 'https://clinicaltrials.gov/api/v2').
    :param parsed_terms: A dictionary containing lists of terms under keys like
                        'conditions', 'interventions', 'outcomes', and 'other_terms'.
    :param page_size: Number of results per page (defaults to 10).
    :param max_results: Optional limit/maximum number of results. (Not officially supported by the API,
                        but included here for illustration.)
    :return: A string representing the full URL with query parameters.
    """
    query_params = []

    # -- 1. Always filter these statuses --
    # The official docs specify pipe-separated values for multiple statuses.
    # We'll percent-encode the pipe character in the final URL.
    query_params.append(
        "filter.overallStatus=NOT_YET_RECRUITING%7CENROLLING_BY_INVITATION%7CRECRUITING%7CAVAILABLE"
    )

    # -- 2. Always sort by relevance --
    query_params.append("sort=%40relevance")  # '@relevance' must be URL-encoded as '%40relevance'

    # -- The rest depends on parsed_terms; these do NOT override the constants above --
    
    conditions_list = parsed_terms.get("conditions", [])
    if conditions_list:
        cond_expr = " OR ".join(conditions_list)
        query_params.append(f"query.cond={cond_expr.replace(' ', '+')}")

    interventions_list = parsed_terms.get("interventions", [])
    if interventions_list:
        intr_expr = " OR ".join(interventions_list)
        query_params.append(f"query.intr={intr_expr.replace(' ', '+')}")

    outcomes_list = parsed_terms.get("outcomes", [])
    if outcomes_list:
        outc_expr = " OR ".join(outcomes_list)
        query_params.append(f"query.outc={outc_expr.replace(' ', '+')}")

    other_terms_list = parsed_terms.get("other_terms", [])
    if other_terms_list:
        terms_expr = " OR ".join(other_terms_list)
        query_params.append(f"query.term={terms_expr.replace(' ', '+')}")

    # Paging / formatting parameters
    query_params.append(f"pageSize={page_size}")
    query_params.append(f"format={output_format}")

    # Construct the full URL
    query_string = "&".join(query_params)
    full_url = f"{base_url}/studies?{query_string}"
    return full_url

def main():
    parser = argparse.ArgumentParser(
        description="Extract key terms from a medical discharge summary, then search clinicaltrials.gov."
    )

    parser.add_argument(
        "-p",
        "--patient-text",
        type=str,
        required=True,
        help="Either the text of the discharge summary or a path to a file containing it."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use for key term extraction."
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="https://clinicaltrials.gov/api/v2",
        help="Base URL for the clinicaltrials.gov API (v2)."
    )

    parser.add_argument(
        "-n",
        "--number-of-trials",
        type=int,
        default=10,
        help="Number of results trials outputted for clinicaltrials.gov queries. Default to top 10."
    )

    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="If provided, only extracts key terms and does NOT query clinicaltrials.gov."
    )
    
    parser.add_argument(
        "-o",
        "--output-format",
        type=str,
        default="csv",
        help="Output format of trials. Please specify between json or csv. Default is csv."
    )

    args = parser.parse_args()

    # Set up OpenAI API key
    openai.api_key = "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"

    # 1. Read the discharge summary. Check if it's a file path.
    if os.path.isfile(args.patient_text):
        with open(args.patient_text, "r", encoding="utf-8") as f:
            discharge_summary_text = f.read()
    else:
        # It's a direct string
        discharge_summary_text = args.patient_text

    print("\nPatient Text: ", discharge_summary_text)
    # 2. Extract key terms
    print("\nExtracting key terms from the patient summary...")
    key_terms_json_str = extract_key_terms(discharge_summary_text, model=args.model)
    
    # Try to parse as JSON
    try:
        key_terms = json.loads(key_terms_json_str)
    except json.JSONDecodeError:
        # If parsing fails, print the raw response
        print("The model did not return valid JSON. Raw output:\n")
        print(key_terms_json_str)
        return

    print("\nKey terms extracted (as JSON):")
    print(json.dumps(key_terms, indent=2))

    # 3. Optionally build a query for clinicaltrials.gov
    if not args.skip_search:
        print("\nBuilding query URL for clinicaltrials.gov v2...")
        query_url = build_ctgov_query(
            base_url=args.base_url,
            parsed_terms=key_terms,
            page_size=args.number_of_trials,
            output_format=args.output_format
        )
        print("Query URL: {}".format(query_url))

        # 4. Query the endpoint
        print("Querying clinicaltrials.gov...")
        response = requests.get(query_url)

        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')

            if 'csv' in content_type.lower():
                print("\nResponse is in CSV format")

                # Decode and parse CSV
                csv_content = io.StringIO(response.text)
                df = pd.read_csv(csv_content)

                print("\nResponse as DataFrame:")
                print(df)  # Show the first few rows

                # Save CSV to file
                csv_filename = "clinicaltrials_response.csv"
                with open(csv_filename, "w", encoding="utf-8") as f:
                    f.write(response.text)

                print(f"\nCSV saved to {os.path.abspath(csv_filename)}")

            elif 'json' in content_type.lower():
                print("\nResponse is in JSON format")

                data = response.json()
                print("\nResponse from clinicaltrials.gov:")
                print(json.dumps(data, indent=2))

                # Save JSON to file
                json_filename = "clinicaltrials_response.json"
                with open(json_filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                print(f"\nJSON saved to {os.path.abspath(json_filename)}")

            else:
                # Neither CSV nor JSON
                raise ValueError(f"Unsupported Content-Type: {content_type}")
        else:
            print(f"Request failed with status code: {response.status_code}")

if __name__ == "__main__":
    main()
