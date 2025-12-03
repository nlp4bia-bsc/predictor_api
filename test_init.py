import argparse

from app.models.predictor import PredictionPipeline

LOCAL_MODEL_PATH = "/PROJECTES/UBIOESDM/PATTERN_PROJECT_BKP_24_10_2025/CogStack-NiFi/ChagasPredictor/saved_models/lstm-attn/11"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the script.")

    parser.add_argument(
        "--case",
        nargs="+",
        help="Visit texts to process"
    )

    parser.add_argument(
        "--dates",
        nargs="+",
        help="String dates corresponding to each visit text ('4Jan2002' format)"
    )

    parser.add_argument(
        "--case-file",
        help="txt file containing the case to execute"
    )
    parser.add_argument(
        "--dates-file",
        help="txt file containing the dates to execute"
    )

    return parser.parse_args()

def main(case, dates):
    pipeline = PredictionPipeline(local_model_path=LOCAL_MODEL_PATH)
    random_footer = {
        "provider_id": "1",
        "person_id": "2",
        "visit_detail_id": "3",
        "note_id": "4",
        "note_type_concept_id": "5",
        "note_datetime": "6",
        "note_title": "7"
    }

    syn_prob, attn_weights = pipeline.predict(case, dates)
    return pipeline.serialize(
        case=case,
        dates=dates,
        syn_prob=syn_prob,
        attn_weights=attn_weights,
        footer=random_footer
    )

if __name__ == "__main__":
    args = parse_arguments()
    if args.case_file:
        with open(args.case_file, encoding="utf-8") as f:
            case = [line.rstrip("\n") for line in f]
    else:
        case = args.case

    if args.dates_file:
        with open(args.dates_file, encoding="utf-8") as f:
            dates = [line.rstrip("\n") for line in f]
    else:
        dates = args.dates

    print(main(case=case, dates=dates))