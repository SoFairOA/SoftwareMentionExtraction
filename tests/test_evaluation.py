# -*- coding: UTF-8 -*-
"""
Created on 10.01.25

:author:     Martin Dočekal
"""
from unittest import TestCase

from datasets import Dataset

from SME.evaluation import normalize_sw_name, normalize_version, LabelsTranslator, dict_of_lists_to_list_of_dicts, \
    align_datasets


class TestNormalizeSWName(TestCase):

    def test_normalize_sw_name(self):
        """
        Test of the normalize_sw_name method.
        """

        self.assertEqual("somesoftware", normalize_sw_name("Some Soft'ware"))


class TestNormalizeVersion(TestCase):

    def test_normalize_version(self):
        """
        Test of the normalize_version method.
        """

        self.assertEqual("123", normalize_version("1.2.3"))
        self.assertEqual("123", normalize_version("build 1.2.3"))
        self.assertEqual("323a", normalize_version("ver 3.2.3A build"))


class TestLabelsTranslator(TestCase):

    def setUp(self):
        self.translator = LabelsTranslator(
            "test_field",
            [
                "B-SOFTWARE", "I-SOFTWARE", "B-VERSION", "I-VERSION", "B-PUBLISHER", "I-PUBLISHER", "B-URL", "I-URL",
            ]
        )

    def test_translate(self):
        """
        Test of the translate method.
        """

        self.assertEqual(
            {
                "labels": [1, 2, 3, 4, 5, 6, 7, 8],
                "test_field": [
                    "B-SOFTWARE", "I-SOFTWARE", "B-VERSION", "I-VERSION", "B-PUBLISHER", "I-PUBLISHER", "B-URL", "I-URL",
                ]
            },
            self.translator(
                {
                    "labels": [1, 2, 3, 4, 5, 6, 7, 8],
                    "test_field": [0, 1, 2, 3, 4, 5, 6, 7],
                }
            )
        )


class TestDictOfListsToListOfDicts(TestCase):

    def test_dict_of_lists_to_list_of_dicts(self):
        """
        Test of the dict_of_lists_to_list_of_dicts method.
        """

        self.assertEqual(
            [
                {"a": 1, "b": 2},
                {"a": 3, "b": 4},
            ],
            dict_of_lists_to_list_of_dicts(
                {
                    "a": [1, 3],
                    "b": [2, 4],
                }
            )
        )


class TestAlignDatasets(TestCase):
    def test_align_datasets_simple(self):
        gold_data = {'id': [1, 2], 'labels': [[1, 2, 3], [4, 5]]}
        results_data = {'id': [1, 2], 'predictions': [[1, 2, 3], [4, 5]]}

        gold_dataset = Dataset.from_dict(gold_data)
        results_dataset = Dataset.from_dict(results_data)

        gold, predictions = align_datasets(
            gold_dataset, results_dataset,
            results_id='id', gold_id='id',
            results_field='predictions', gold_field='labels'
        )
        self.assertEqual(gold, [[1, 2, 3], [4, 5]])
        self.assertEqual(predictions, [[1, 2, 3], [4, 5]])

    def test_align_datasets_dict_of_lists(self):
        gold_data = {
            'id': [1, 2],
            'labels': [{'a': [1, 2], 'b': [3, 4]}, {'a': [5, 6], 'b': [7, 8]}]
        }
        results_data = {
            'id': [1, 2],
            'predictions': [{'a': [1, 2], 'b': [3, 4]}, {'a': [5, 6], 'b': [7, 8]}]
        }

        gold_dataset = Dataset.from_dict(gold_data)
        results_dataset = Dataset.from_dict(results_data)

        gold, predictions = align_datasets(
            gold_dataset, results_dataset,
            results_id='id', gold_id='id',
            results_field='predictions', gold_field='labels'
        )
        expected_gold = [
            [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}],
            [{'a': 5, 'b': 7}, {'a': 6, 'b': 8}]
        ]
        self.assertEqual(gold, expected_gold)
