# Classification Inference API

This is a Flask API that performs bert classification using any bert model of your preference. The model relies on having sotred the pretrained model in a cache direccotry  beforehand and having the finetuned model in a local directory. 

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

2. Save the model in a cache directory, which by default, is in **app/models/hugginface_mirror**. You can change this path to be your hugginface cache dir by changing the path in **app/models/config/model_config.py** file.

To do so from the terminal, go to the **app/models** directory and execute:
```bash
huggingface-cli download <pretrained_model_hf_path> --cache-dir ./huggingface_mirror
```
To do so using python you can execute
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dmis-lab/biobert-base-cased-v1.1", 
    cache_dir="./huggingface_mirror"
)
```

3. Load you model --as a state dictionary-- into the app directory and reference the path in the aforementioned config path.
**IMPORTANT:** This model must have been finetuned using the same model constructor  as referenced in the **app/models/model_baseline.py** file, in this case `BinaryBERT`

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
      "text": "A 64-year-old woman was admitted to our hospital because of dyspnea and chest pain for 3 month.",
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
        "class_logits":[0.9320483207702637,0.022358665242791176,0.028913727030158043,0.01667933166027069],
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
            "text":"A 74-year-old woman was admitted to our hospital because of dyspnea and chest pain for 1 month."
            }
        },
        "nlp_service_info":{"service_app_name":"NLP Classifier",
        "service_language":"es",
        "service_model":"PredictionPipeline",
        "service_version":"1.0"}}
```
2. /process_bulk - Process Multiple Texts
Method: POST

This endpoint processes multiple texts at once, applying NER using the dictionary model to each text.

Example Request

```bash
curl --location 'http://0.0.0.0:8002/process_bulk' \
--header 'Content-Type: application/json' \
--data '{
  "content": [
    {
      "text": "A 74-year-old woman was admitted to our hospital because of dyspnea and chest pain for 1 month.",
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
      "text": "A 6-year-old man was admitted to our hospital because of metastatic tumor.",
      "footer": {
        "provider_id": "8",
        "person_id": "9",
        "visit_detail_id": "10",
        "note_id": "11",
        "note_type_concept_id": "12",
        "note_datetime": "13",
        "note_title": "14"
      }

    }
  ]
}'

```
Example Response

```json
[
    {
        "nlp_output":{
            "class_logits":[0.9320483207702637,0.022358665242791176,0.028913727030158043,0.01667933166027069],
            "processing_success":true,
            "record_metadata":{
                "admission_id":"3",
                "clinical_site_id":"1",
                "deidentification_pipeline_name":"",
                "deidentification_pipeline_version":"",
                "deidentified":"no",
                "nlp_processing_date":"2025-06-18T13:34:45.985687",
                "nlp_processing_pipeline_name":"PredictionPipeline",
                "nlp_processing_pipeline_version":"1.0",
                "patient_id":"2",
                "record_character_encoding":"UTF-8",
                "record_creation_date":"6",
                "record_extraction_date":"2025-06-18T13:34:45.985685",
                "record_format":"json",
                "record_id":"4",
                "record_lastupdate_date":"2025-06-18T13:34:45.985676",
                "record_type":"5",
                "report_language":"es",
                "report_section":"7",
                "text":"A 74-year-old woman was admitted to our hospital because of dyspnea and chest pain for 1 month."
            }
        },
        "nlp_service_info":{
            "service_app_name":"NLP Classifier",
            "service_language":"es",
            "service_model":"PredictionPipeline",
            "service_version":"1.0"
        }
    },
    {
        "nlp_output":{
            "class_logits":[0.8947045207023621,0.023609697818756104,0.04161892458796501,0.040066950023174286],
            "processing_success":true,
            "record_metadata":{
                "admission_id":"10",
                "clinical_site_id":"8",
                "deidentification_pipeline_name":"",
                "deidentification_pipeline_version":"",
                "deidentified":"no",
                "nlp_processing_date":"2025-06-18T13:34:46.082486",
                "nlp_processing_pipeline_name":"PredictionPipeline",
                "nlp_processing_pipeline_version":"1.0",
                "patient_id":"9",
                "record_character_encoding":"UTF-8",
                "record_creation_date":"13",
                "record_extraction_date":"2025-06-18T13:34:46.082484",
                "record_format":"json",
                "record_id":"11",
                "record_lastupdate_date":"2025-06-18T13:34:46.082475",
                "record_type":"12",
                "report_language":"es",
                "report_section":"14",
                "text":"A 6-year-old man was admitted to our hospital because of metastatic tumor."
            }
        },
        "nlp_service_info":{
            "service_app_name":"NLP Classifier",
            "service_language":"es",
            "service_model":"PredictionPipeline",
            "service_version":"1.0"
        }
    }
]
```

## To stop the service, use:

```bash
docker-compose down
```

## Trouble Shooting
When calling the endpoint, if an internal error occurs it is much more complicated to debug than if the inference pipeline is executed directly. To facilitate debugging in case of errors, I have added a testing file called `test_init.py` which you can call alongside a sample text like follows:

```bash
python test_init.py --text "sample text"
```
Bear in mind you will have to create a virtual environment with the modules defined in `requirements.txt` installed
