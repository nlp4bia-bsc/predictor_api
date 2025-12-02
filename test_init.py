import argparse

from app.models.predictor import PredictionPipeline

LOCAL_MODEL_PATH = "test"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the script.")

    parser.add_argument(
        "--cases",
        nargs="+",
        help="Visit texts to process"
    )

    parser.add_argument(
        "--dates",
        nargs="+",
        help="String dates corresponding to each visit text ('4Jan2002' format)"
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
        syn_prob=syn_prob,
        attn_weights=attn_weights,
        footer=random_footer
    )

if __name__ == "__main__":
    args = parse_arguments()
    print(main(case=args.cases, dates=args.dates))