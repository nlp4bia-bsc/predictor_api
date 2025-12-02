from abc import ABC, abstractmethod
from datetime import datetime
import json

class ModelClass(ABC):
    @abstractmethod
    def predict(self, case: list[str], visit_times: list[str]) -> tuple[float, list[float]]:
        pass

    def serialize(self, case: list[str], syn_prob: float, attn_weights: list[float], footer: dict):
        """This function implements the Common Data Model v2"""
        output = {
            "nlp_output": {
                "record_metadata": {
                    "clinical_site_id": footer['provider_id'],
                    "patient_id": footer['person_id'],
                    "admission_id": footer['visit_detail_id'],
                    "record_id": footer['note_id'],
                    "record_type": footer['note_type_concept_id'],
                    "record_format": "json",
                    "record_creation_date": footer['note_datetime'],
                    "record_lastupdate_date": datetime.now().isoformat(),
                    "record_character_encoding": "UTF-8",
                    "record_extraction_date": datetime.now().isoformat(),
                    "report_section": footer['note_title'],
                    "report_language": "es",
                    "deidentified": "no",
                    "deidentification_pipeline_name": "",
                    "deidentification_pipeline_version": "",
                    "case": case,
                    "nlp_processing_date": datetime.now().isoformat(),
                    "nlp_processing_pipeline_name": self.__class__.__name__,
                    "nlp_processing_pipeline_version": "1.0",
                },
                "syntomatic_probability": syn_prob,
                "attention_weights": attn_weights
            },
            "nlp_service_info": {
                "service_app_name": "NLP Chagas Prediction",
                "service_language": "es",
                "service_version": "1.0",
                "service_model": self.__class__.__name__
            }
        }
        output["nlp_output"]["processing_success"] = True
        return output
