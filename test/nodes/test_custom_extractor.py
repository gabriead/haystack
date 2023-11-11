import unittest
import pytest
from haystack.nodes.extractor.entity_extractor import EntityExtractor
from haystack.preview import Pipeline, Document
from typing import List, Dict

class EntityExtractorTest(unittest.TestCase):

    def test_init_model(self):
        entity_extractor = EntityExtractor()
        self.assertTrue(entity_extractor.model_name_or_path,"dslim/bert-base-NER")

    def test_entity_extraction_locations_metadata(self):
        ner = EntityExtractor(
            model_name_or_path="dslim/bert-base-NER",
            max_seq_len=6,
            aggregation_strategy="simple",
        )

        text = "I live in Berlin and I am traveling to Kuwait."
        output = ner.extract(text)
        output = [x.pop("score") for x in output]

        self.assertTrue(len(output), 2)
        self.assertTrue(type(output) , List)
        self.assertTrue(output[0], {"entity_group": "LOC", "word": "Berlin", "start": 10, "end": 16})
        self.assertTrue(output[1], {"entity_group": "LOC", "word": "Kuwait", "start": 39, "end": 45})

    def test_entity_extractor_in_pipeline(self):
        entity_pipeline = Pipeline()
        entity_pipeline.add_component(instance=EntityExtractor(), name="entity_extractor")

        documents = [
            Document(text="My name is Jean and I live in Paris."),
            Document(text="My name is Mark and I live in Berlin."),
            Document(text="My name is Giorgio and I live in Rome."),
        ]

        entities = entity_pipeline.run({"entity_extractor": {"documents":documents}})
        self.assertTrue(entities["entity_extractor"]["documents"][0].metadata["entities"][1] , {"entity_group": "LOC", "word": "Paris", "start": 30, "end": 35, 'score': 0.9995563903580217})
        self.assertTrue(entities["entity_extractor"]["documents"][0].metadata["entities"][0] , {"entity_group": "PER", "word": "Jean", "start": 11, "end": 15, 'score': 0.9990130414246812})

        self.assertTrue(entities["entity_extractor"]["documents"][1].metadata["entities"][1] , {"entity_group": "LOC", "word": "Berlin", "start": 30, "end": 36, 'score': 0.9995246222007287})
        self.assertTrue(entities["entity_extractor"]["documents"][1].metadata["entities"][0] , {"entity_group": "PER", "word": "Mark", "start": 11, "end": 15, 'score': 0.9992774693192789})

        self.assertTrue(entities["entity_extractor"]["documents"][2].metadata["entities"][1] , {"entity_group": "LOC", "word": "Rome", "start": 33, "end": 37, 'score': 0.9994580319420049})
        self.assertTrue(entities["entity_extractor"]["documents"][2].metadata["entities"][0] , {"entity_group": "PER", "word": "Giorgio", "start": 11, "end": 18, 'score': 0.9986354950934884})