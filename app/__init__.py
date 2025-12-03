from flask import Flask, request, jsonify
from flasgger import Swagger

from app.models.predictor import PredictionPipeline

app = Flask(__name__)
swagger = Swagger(app)

LOCAL_MODEL_PATH = "/PROJECTES/UBIOESDM/PATTERN_PROJECT_BKP_24_10_2025/CogStack-NiFi/ChagasPredictor/saved_models/lstm-attn/11"

pipeline = PredictionPipeline(local_model_path=LOCAL_MODEL_PATH)

def infer_case(content: dict):
    if 'case' not in content or 'dates' not in content:
        return jsonify({"error": "content must be contain 'dates' and 'text' keys"}), 400
    
    case = content.get('case', None)
    dates = content.get('dates', None)

    if case is None or not isinstance(case, list):
        return jsonify({"error": "'case' must be a list of strings"}), 400
    if dates is not None or not isinstance(dates, list):
        return jsonify({"error": "'dates' must be a list of DDMonYYYY format date strings or nulls"}), 400

    if len(case) != len(dates):
        return jsonify({"error": f"'case' ({len(case)} items) and 'dates' ({len(dates)} items) must have the same length"}), 400

    syn_prob, attn_weights = pipeline.predict(case, dates)
    footer = content.get('footer', {})
    serialized_output = pipeline.serialize(
        case=case,
        dates=dates,
        syn_prob=syn_prob,
        attn_weights=attn_weights,
        footer=footer
    )
    return serialized_output

@app.route('/process_text', methods=['POST'])
def process_text():
    """
    Process text using the trained classification model
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: text_input
          required:
            - case
            - dates
          properties:
            case:
              type: list
              description: The visit text entries to process
            dates:
              type: list
              description: The visit dates corresponding to each text entry
    responses:
      200:
        description: Processed text with classification results
    """
    data = request.json
    if not isinstance(data, dict):
        return jsonify({"error": "Input must be a dictionary"}), 400

    if not "content" in data.keys():
        return jsonify({"error": "Input must be a dictionary with 'content' key"}), 400

    content = data["content"]

    result = infer_case(content)
    return jsonify(result)

@app.route('/process_bulk', methods=['POST'])
def process_bulk():
    """
    Process multiple texts using the trained classification model
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: bulk_input
          type: array
          items:
            type: object
            required:
              - text
            properties:
              text:
                type: string
                description: The text to process
    responses:
      200:
        description: Processed texts with NER annotations
    """
    data = request.json

    if not isinstance(data, dict):
        return jsonify({"error": "Input must be a dictionary"}), 400

    if not "content" in data.keys():
        return jsonify({"error": "Input must be a dictionary with 'content' key"}), 400

    content_list = data["content"]

    if not isinstance(data, list):
        return jsonify({"error": "Content in input must be a list of objects"}), 400

    results = []
    for content in content_list:
        result = infer_case(content)
        results.append(result)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
