# Prediction Inference API

This is a Flask API that performs prediction over many spanish clinical visits using any roberta model of your preference along with a LSTM, attention and classification head modules. The model relies on having trained the finetuned model beforehand and have it stored in a local directory (along with its config and tokenizer). An example can be found in [this huggingface model card](https://duckduckgo.com) 

## Prerequisites

Before starting, make sure you have Docker and Docker Compose installed on your system.

* Docker
* Docker Compose
## Instructions to Start the Service

1. Clone the repository
First, clone the repository to your local machine:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Save the model in a local directory.

To do so from the terminal, go to the **app/models** directory and execute:
```bash
huggingface-cli download <pretrained_model_hf_path> --local-dir your/model/folder
```
To do so using python you can execute
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dmis-lab/biobert-base-cased-v1.1", 
    local_dir="your/model/folder"
)
```
You must configure this path in the **app/__init__.py**, in the `python LOCAL_MODEL_PATH` variable.

3. Load you model --as a huggigface PreTrainedModel-- into the app directory and reference the path in the aforementioned config path.
**IMPORTANT:** This model must have been finetuned using the same model constructor as referenced in the **app/models/model_baseline.py** file, in this case `LSTMBERT`

4. Build and Run the Docker Container
Build and run the service using Docker Compose:

```bash
docker-compose up --build
```
This will start the service on port 8002. The port can be changed in the docker-compose file.

5. Verify the Service is Running
You can open your browser or run a curl request to http://localhost:8002 to ensure the service is up and running.

## Available Endpoints

1. /process_text - Process a Single Text
Method: POST

This endpoint processes a single text and applies NER using the dictionary model.

Example Request

```bash
curl --location 'http://0.0.0.0:8002/process_text' \
--header 'Content-Type: application/json' \
--data '{
  "content":{
      "case": ["Paciente con fiebre y dolor de cabeza.", "Se observa inflamación en las articulaciones."],
      "dates": ['10Jan2024', '9Apr2024'],
      "footer": {
        "provider_id": "1",
        "person_id": "2",
        "visit_detail_id": "3",
        "note_id": "4",
        "note_type_concept_id": "5",
        "note_datetime": "6",
        "note_title": "7"
      }
  }
}'
```
Example Response

```json
{
    "nlp_output":{
        "syntomatic_probability": 0.5387078523635864,
        "attention_weights": [0.49486151337623596, 0.5051384568214417],
        "processing_success":true,
        "record_metadata":{
            "admission_id":"3",
            "clinical_site_id":"1",
            "deidentification_pipeline_name":"",
            "deidentification_pipeline_version":"",
            "deidentified":"no",
            "nlp_processing_date":"2025-06-18T13:19:48.677977",
            "nlp_processing_pipeline_name":"PredictionPipeline",
            "nlp_processing_pipeline_version":"1.0",
            "patient_id":"2",
            "record_character_encoding":"UTF-8",
            "record_creation_date":"6",
            "record_extraction_date":"2025-06-18T13:19:48.677975",
            "record_format":"json",
            "record_id":"4",
            "record_lastupdate_date":"2025-06-18T13:19:48.677966",
            "record_type":"5",
            "report_language":"es",
            "report_section":"7",
            "case":["Paciente con fiebre y dolor de cabeza.", "Se observa inflamación en las articulaciones."],
            "dates": ['10Jan2024', '9Apr2024']
            }
        },
        "nlp_service_info":{"service_app_name":"NLP Classifier",
        "service_language":"es",
        "service_model":"PredictionPipeline",
        "service_version":"1.0"}}
```
2. /process_bulk - Process Multiple Texts
Method: POST

This endpoint processes multiple cases at once.

Example Request

```bash
curl --location 'http://0.0.0.0:8002/process_bulk' \
--header 'Content-Type: application/json' \
--data '{
  "content": [
    {
      "case": ["Paciente con fiebre y dolor de cabeza.", "Se observa inflamación en las articulaciones."],
      "dates": ['10Jan2024', '9Apr2024'],
      "footer": {
        "provider_id": "1",
        "person_id": "2",
        "visit_detail_id": "3",
        "note_id": "4",
        "note_type_concept_id": "5",
        "note_datetime": "6",
        "note_title": "7"
      }
    },
    {
      "case": ["Paciente con fiebre y dolor de cabeza.", "Se observa inflamación en las articulaciones."],
      "dates": ['10Jan2024', '9Apr2024'],
      "footer": {
        "provider_id": "1",
        "person_id": "2",
        "visit_detail_id": "3",
        "note_id": "4",
        "note_type_concept_id": "5",
        "note_datetime": "6",
        "note_title": "7"
      }
    }
  ]
}'

```
Example Response

```json
[
    {}
]
```

## To stop the service, use:

```bash
docker-compose down
```

## Trouble Shooting
When calling the endpoint, if an internal error occurs it is much more complicated to debug than if the inference pipeline is executed directly. To facilitate debugging in case of errors, I have added a testing file called `test_init.py` which you can call alongside a sample text like follows:

```bash
python test_init.py --cases "Paciente con fiebre y dolor de cabeza." "Se observa inflamación en las articulaciones." --dates "10Jan2024" "9Apr2024"
```
Bear in mind you will have to create a virtual environment with the modules defined in `app/requirements.txt` installed in order to execute this python file.

Additionally, I have also provided a jupyter notebook to debug and understand the flow of data and the model loading and unloading. (`test.ipynb`)
