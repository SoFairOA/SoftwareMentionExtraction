import copy
import json
import re
import string
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import evaluate
import datasets
from datasets import load_dataset

remove_punctuation_table = str.maketrans(string.punctuation, " " * len(string.punctuation))


def normalize_sw_name(sw_name: str) -> str:
    """
    Normalizes software name.
    :param sw_name: Software name.
    :return: Normalized software name.
    """
    return sw_name.lower().translate(remove_punctuation_table).replace(" ", "")


def normalize_version(version: str) -> str:
    """
    Normalizes version.
    :param version: Version.
    :return: Normalized version.
    """

    version = version.lower()

    version = re.sub(r"\b(v|ver|version|release|rel|build|b|patch|p|update|u|edition|ed|e|)\b", "", version)

    return version.translate(remove_punctuation_table).replace(" ", "")


class LabelsTranslator:
    """
    Functor for translating labels in integer format to string format.
    """

    def __init__(self, field: str, mapping: list[str]):
        """
        Initializes the translator.

        :param field: Field name of the labels in the dataset.
        :param mapping: Mapping of the labels.
        """
        self.field = field
        self.mapping = mapping

    def __call__(self, example: dict) -> dict:
        """
        Translates labels.

        :param example: Example from the dataset.
        :return: Translated example.
        """
        example[self.field] = list(map(lambda x: self.mapping[x], example[self.field]))
        return example


def load_and_check_datasets(results_path: str, gold_path: str, results_id: str, gold_id: str, results_field: str,
                            gold_field: str, split: str = "test", config: str = None,
                            hf_cache: str = None, allow_subset: bool = False):
    """
    Load and check the datasets.

    :param results_path: Path to the results file.
    :param gold_path: Path to the gold file.
    :param results_id: Field name with unique identifier in the results dataset.
    :param gold_id: Field name with unique identifier in the gold dataset.
    :param results_field: Field name with results in the results dataset.
    :param gold_field: Field name with ground truth in the gold dataset.
    :param split: Split of the gold dataset.
    :param config: Config name of the gold dataset. Is also used for determining type of results field.
    :param hf_cache: Path to the Hugging Face cache.
    :param allow_subset: Allow subset of the gold dataset.
    :return: Gold and results datasets.
    """

    extension = Path(results_path).suffix.lstrip(".")
    # load
    try:
        data_type = {
            "json": "json",
            "csv": "csv",
            "tsv": "tsv",
            "jsonl": "json",
        }[extension]
    except KeyError:
        print(f"Unknown file format ({extension}). Exiting.")
        exit(1)

    results_dataset = load_dataset(data_type, data_files=results_path)["train"]
    if data_type != "json":
        # hopped that it can be solved by defining Features, however it seemed that in that case it is necessary to define all fields
        def convert_json_fields(example):
            example[results_field] = json.loads(example[results_field])
            return example

        results_dataset = results_dataset.map(convert_json_fields)

    gold_dataset = load_dataset(gold_path, split=split, name=config, cache_dir=hf_cache)

    results_dataset = results_dataset.select_columns([results_id, results_field])
    gold_dataset = gold_dataset.select_columns([gold_id, gold_field])

    # check
    results_ids = set(results_dataset[results_id])
    gold_ids = set(gold_dataset[gold_id])

    duplicity_check = False
    # duplicity check
    if len(results_ids) != len(results_dataset[results_id]):
        print("Results contain duplicities.")
        duplicity_check = True
        for i in results_ids:
            if results_dataset[results_id].count(i) > 1:
                print(f"ID {i} is duplicated.")

    if len(gold_ids) != len(gold_dataset[gold_id]):
        print("Gold contain duplicities.")
        duplicity_check = True
        for i in gold_ids:
            if gold_dataset[gold_id].count(i) > 1:
                print(f"ID {i} is duplicated.")

    if duplicity_check:
        print("Exiting due to duplicities.")
        exit(1)

    if results_ids != gold_ids:
        if (not allow_subset or not results_ids.issubset(gold_ids)):
            print("Results and gold are not aligned. Exiting.")
            exit(1)

        gold_dataset = gold_dataset.filter(lambda x: x[gold_id] in results_ids)

    return gold_dataset, results_dataset


def dict_of_lists_to_list_of_dicts(d: dict) -> list[dict]:
    """
    Converts dictionary of lists to list of dictionaries.

    It is assumed that all lists have the same length.

    :param d: Dictionary of lists.
    :return: List of dictionaries.
    """

    first_length = len(d[list(d.keys())[0]])
    assert all(len(v) == first_length for v in d.values()), "All lists must have the same length."

    return [{k: v[i] for k, v in d.items()} for i in range(first_length)]


def align_datasets(gold_dataset, results_dataset, results_id: str, gold_id: str, results_field: str, gold_field: str) \
        -> tuple[list, list]:
    """
    Align the results and gold datasets and returns sequence of predictions and ground truth labels.

    :param gold_dataset: gold dataset
    :param results_dataset: results dataset
    :param results_id: Field name with unique identifier in the results dataset.
    :param gold_id: Field name with unique identifier in the gold dataset.
    :param results_field: Field name with results in the results dataset.
    :param gold_field: Field name with ground truth in the gold dataset.
    :return:
        ground truth labels
        predictions
    """

    gold, results = [], []
    results_mapping = {example[results_id]: example[results_field] for example in results_dataset}
    for gold_example in gold_dataset:
        gold.append(
            gold_example[gold_field] if isinstance(gold_example[gold_field], list) else dict_of_lists_to_list_of_dicts(
                gold_example[gold_field])
        )

        r = results_mapping[gold_example[gold_id]]
        if isinstance(r, dict):
            r = dict_of_lists_to_list_of_dicts(r)
        results.append(r)
    return gold, results


def convert_numpy_types(d: dict) -> dict:
    """
    Convert all numpy types to python types recursively.

    :param d: Dictionary to convert.
    :return: Converted dictionary.
    """
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = convert_numpy_types(v)
        elif hasattr(v, "item"):
            d[k] = v.item()
    return d


def map_seq_labels(path_to_mapping: str, labels: list[list[str]]) -> list[list[str]]:
    """
    Maps sequence labels.

    :param path_to_mapping: Path to the mapping file.
    :param labels: Labels to map.
    :return: Mapped labels.
    """
    with open(path_to_mapping, "r") as f:
        mapping = json.load(f)

    return [[mapping.get(label, label) for label in seq] for seq in labels]


def sequence_labeling(args):
    """
    Evaluation of sequence labeling.

    :param args: User arguments.
    """

    # load
    gold_dataset, results_dataset = load_and_check_datasets(args.results, args.gold, args.results_id, args.id,
                                                            args.prediction_field, args.gt_field, args.split,
                                                            args.config, args.hf_cache, args.allow_subset)

    label_names = None
    # translate the labels
    if not args.disable_translate and (hasattr(gold_dataset.features[args.gt_field], "feature") and hasattr(
            gold_dataset.features[args.gt_field].feature, "names")):
        label_names = gold_dataset.features[args.gt_field].feature.names

        if args.gold_mapping is not None:
            with open(args.gold_mapping, "r") as f:
                mapping = json.load(f)

            label_names = [mapping.get(label, label) for label in label_names]

        # check length of samples
        for i, (gold_example, results_example) in enumerate(zip(gold_dataset, results_dataset)):
            if len(gold_example[args.gt_field]) != len(results_example[args.prediction_field]):
                print(
                    f"Length of the prediction and ground truth is different for sample {i}. Prediction: {len(results_example[args.prediction_field])}, Ground truth: {len(gold_example[args.gt_field])}")

        results_dataset = results_dataset.map(
            LabelsTranslator(args.prediction_field, label_names),
            load_from_cache_file=False,
            keep_in_memory=True
        )
        # for some reason it was necessary to specify this casting
        gold_features = gold_dataset.features.copy()
        gold_features[args.gt_field] = datasets.Sequence(datasets.Value("string"))

        gold_dataset = gold_dataset.map(
            LabelsTranslator(args.gt_field, label_names),
            load_from_cache_file=False,
            keep_in_memory=True,
            features=gold_features
        )

        # check length of samples
        for i, (gold_example, results_example) in enumerate(zip(gold_dataset, results_dataset)):
            if len(gold_example[args.gt_field]) != len(results_example[args.prediction_field]):
                print(
                    f"2 Length of the prediction and ground truth is different for sample {i}. Prediction: {len(results_example[args.prediction_field])}, Ground truth: {len(gold_example[args.gt_field])}")

    # align the results and gold
    gold, results = align_datasets(gold_dataset, results_dataset, args.results_id, args.id, args.prediction_field,
                                   args.gt_field)

    # evaluate
    seqeval = evaluate.load("seqeval")

    if args.results_mapping is not None:
        results = map_seq_labels(args.results_mapping, results)

    if args.gold_mapping is not None:
        gold = map_seq_labels(args.gold_mapping, gold)

    for i, (x, y) in enumerate(zip(results, gold)):
        if len(x) != len(y):
            print(
                f"Length of the prediction and ground truth is different for sample {i}. Prediction: {len(x)}, Ground truth: {len(y)}")

    if args.only_with_annotation:
        new_results = []
        new_gold = []
        for r, g in zip(results, gold):
            if (label_names is not None and any(x != label_names[0] for x in g)) or (label_names is None and any(x != 0 for x in g)):
                new_results.append(r)
                new_gold.append(g)

        print(f"Filtered only samples with at least one ground truth annotation. After filtering there are {len(new_gold)} samples out of {len(gold)}")
        results = new_results
        gold = new_gold

    eval_res = seqeval.compute(predictions=results, references=gold)
    # convert all numpy types to python types, so we can serialize it to json
    eval_res = convert_numpy_types(eval_res)
    print(json.dumps(eval_res))


def map_software_attribute_names(path_to_mapping: str, software: list[list[dict]]) -> list[list[dict]]:
    """
    Maps software attributes.

    :param path_to_mapping: Path to the mapping file.
    :param software: Software attributes to map.
    :return: Mapped software attributes.
    """
    with open(path_to_mapping, "r") as f:
        mapping = json.load(f)

    new_software = []

    for sample in software:
        new_sample = []
        for soft in sample:
            new_soft = {}
            for k, v in soft.items():
                mapped_k = mapping.get(k, k)
                if mapped_k is None:
                    continue

                if mapped_k in new_soft:
                    new_soft[mapped_k].extend(v)
                else:
                    new_soft[mapped_k] = v

            new_sample.append(new_soft)

        new_software.append(new_sample)

    return new_software


def count_statistics_for_doc_lvl(predictions: list[list[Any]], labels: list[list[Any]]) -> dict[str, int]:
    """
    Provides statistics about number of predictions, number of labels, number of empty samples, ...

    :param predictions: list of preidctions for each sample
    :param labels: list of labels for each sample
    :return: statistics
    """

    return {
        "number_of_predictions": sum(len(s) for s in predictions),
        "number_of_labels": sum(len(s) for s in labels),
        "samples_without_predictions": sum(len(s) == 0 for s in predictions),
        "samples_without_labels": sum(len(s) == 0 for s in labels),
    }


def document_level(args):
    """
    Document level evaluation.
    :param args: User arguments.
    """

    # load
    gold_dataset, results_dataset = load_and_check_datasets(args.results, args.gold, args.results_id, args.id,
                                                            args.prediction_field, args.gt_field, args.split,
                                                            args.config, args.hf_cache, args.allow_subset)

    # align the results and gold
    gold, results = align_datasets(gold_dataset, results_dataset, args.results_id, args.id, args.prediction_field,
                                   args.gt_field)

    if args.results_mapping is not None:
        results = map_software_attribute_names(args.results_mapping, results)

    if args.gold_mapping is not None:
        gold = map_software_attribute_names(args.gold_mapping, gold)

    # parse software names
    gold_software_names, results_software_names = [], []

    gold_properties = {
        "version": [],
        "publisher": [],
        "url": [],
        "language": []
    }
    results_properties = copy.deepcopy(gold_properties)
    gold_properties_independent, results_properties_independent = copy.deepcopy(gold_properties), copy.deepcopy(
        gold_properties)

    norm_for_properties = {
        "version": normalize_version,
        "publisher": normalize_sw_name,
        "url": lambda x: x,
        "language": lambda x: x.lower()
    }

    skipped_empty = 0
    for i in range(len(gold)):
        if args.only_with_annotation and len(gold[i]) == 0:
            skipped_empty += 1
            continue
        if args.only_with_mention and len(gold[i]) == 0 and len(results[i]) == 0:
            skipped_empty += 1
            continue

        g_names_normalized = [normalize_sw_name(x["name"]) for x in gold[i]]
        gold_software_names.append(g_names_normalized)
        results_software_names.append([normalize_sw_name(x["name"]) for x in results[i]])

        for prop in gold_properties:
            prop_norm = norm_for_properties[prop] if args.normalize_properties else lambda x: x

            if len(gold[i]) > 0 and prop not in gold[i][0]:
                gold_properties[prop].append([])
                gold_properties_independent[prop].append([])
            else:
                gold_properties[prop].append([
                    f"{s_name} {prop_norm(p)}" for i_s, s_name in enumerate(g_names_normalized) for p in
                    gold[i][i_s][prop]
                ])
                gold_properties_independent[prop].append([
                    prop_norm(p) for soft in gold[i] for p in soft[prop]
                ])

            results_properties[prop].append([
                f"{normalize_sw_name(soft['name'])} {prop_norm(p)}" for soft in results[i] for p in soft[prop]
            ])

            results_properties_independent[prop].append([
                prop_norm(p) for soft in results[i] for p in soft[prop]
            ])

    # evaluate
    doc_level = evaluate.load("mdocekal/multi_label_precision_recall_accuracy_fscore")
    doc_level.info.features = datasets.Features({
        # we need to specify this, because the HF evaluator is not able to infer data types correctly when the first example is empty
        'predictions': datasets.Sequence(datasets.Value('string')),
        'references': datasets.Sequence(datasets.Value('string')),
    })

    if args.only_with_annotation:
        print(
            f"Filtered only samples with at least one ground truth annotation. After filtering there are {len(gold_software_names)} samples out of {len(gold_software_names) + skipped_empty}")

    if args.only_with_mention:
        print(
            f"Filtered only samples with at least one mention. After filtering there are {len(gold_software_names)} samples out of {len(gold_software_names) + skipped_empty}")

    print(f"Number of samples: {len(gold_software_names)}")
    eval_res = doc_level.compute(predictions=results_software_names, references=gold_software_names)
    print("Document level software mentions extraction evaluation:")
    eval_res.update(count_statistics_for_doc_lvl(results_software_names, gold_software_names))
    print(json.dumps(eval_res))

    # evaluate properties
    for prop in gold_properties:
        print(f"Document level {prop} extraction evaluation:")
        eval_res = doc_level.compute(predictions=results_properties[prop], references=gold_properties[prop])
        eval_res.update(count_statistics_for_doc_lvl(results_properties[prop], gold_properties[prop]))
        print("\t" + json.dumps(eval_res))
        print("\tIndependent evaluation:")
        eval_res = doc_level.compute(predictions=results_properties_independent[prop],
                                     references=gold_properties_independent[prop])
        eval_res.update(
            count_statistics_for_doc_lvl(results_properties_independent[prop], gold_properties_independent[prop]))
        print("\t\t" + json.dumps(eval_res))


def intent(args):
    """
    Evaluation of intent classification.

    :param args: User arguments.
    """
    # load
    gold_dataset, results_dataset = load_and_check_datasets(args.results, args.gold, args.results_id, args.id,
                                                            args.prediction_field, args.gt_field, args.split,
                                                            args.config, args.hf_cache, args.allow_subset)

    # align the results and gold
    gold, results = align_datasets(gold_dataset, results_dataset, args.results_id, args.id, args.prediction_field,
                                   args.gt_field)

    acc = evaluate.load("accuracy")
    eval_res = acc.compute(predictions=results, references=gold)
    print(f"accuracy evaluation:\t{eval_res}")

    # evaluate
    for metric_name in ["precision", "recall", "f1"]:
        metric = evaluate.load(metric_name)

        eval_res = metric.compute(predictions=results, references=gold, average=None)

        print(f"{metric_name} evaluation:")
        print(f"\tmacro average: {eval_res[metric_name].mean()}")
        for class_name, value in zip(gold_dataset.features[args.gt_field].names, eval_res[metric_name]):
            print(f"\t{class_name}: {value}")


def main():
    parser = ArgumentParser(
        description="Script for evaluation of software mentions extraction. The results are printed to stdout.")
    subparsers = parser.add_subparsers()

    sequence_labeling_parser = subparsers.add_parser("sequence_labeling", help="Sequence labeling evaluation.")
    sequence_labeling_parser.add_argument("results", help="Path to the results file.")
    sequence_labeling_parser.add_argument("--gold",
                                          help="Name/path of the gold Hugging Face dataset.",
                                          default="SoFairOA/softcite_dataset")
    sequence_labeling_parser.add_argument("-p", "--prediction_field",
                                          help="Field name with results in the results dataset.", default="labels")
    sequence_labeling_parser.add_argument("-g", "--gt_field", help="Field name with ground truth in the gold dataset.",
                                          default="labels")
    sequence_labeling_parser.add_argument("-s", "--split", help="Split of the gold dataset.", default="test")
    sequence_labeling_parser.add_argument("-c", "--config", help="Config name of the gold dataset.",
                                          default="documents")
    sequence_labeling_parser.add_argument("-i", "--id",
                                          help="Name of field with unique identifier in the gold dataset. Is used to make sure that the results and gold are aligned.",
                                          default="id")
    sequence_labeling_parser.add_argument("-r", "--results_id",
                                          help="Name of field with unique identifier in the results dataset. Is used to make sure that the results and gold are aligned.",
                                          default="id")
    sequence_labeling_parser.add_argument("--hf_cache", help="Path to the Hugging Face cache.", default=None)
    sequence_labeling_parser.add_argument("--disable_translate", help="Disables translation of the results and gold.",
                                          action="store_true")
    sequence_labeling_parser.add_argument("--allow_subset", help="Allow evaluation of subset of the gold dataset.",
                                          action="store_true")
    sequence_labeling_parser.add_argument("--results_mapping",
                                          help="Mapping used to transform labels (json file). Could be used to convert labels from different datasets to the same format.",
                                          default=None)
    sequence_labeling_parser.add_argument("--gold_mapping",
                                          help="Mapping used to transform labels (json file). Could be used to convert labels from different datasets to the same format.",
                                          default=None)
    sequence_labeling_parser.add_argument("--only_with_annotation",
                                          help="Evaluates only samples with at least one mention in ground truth.",
                                          action="store_true")
    sequence_labeling_parser.set_defaults(func=sequence_labeling)

    document_level_parser = subparsers.add_parser("document_level", help="Document level evaluation.")
    document_level_parser.add_argument("results", help="Path to the results file.")
    document_level_parser.add_argument("--gold",
                                       help="Name/path of the gold Hugging Face dataset.",
                                       default="SoFairOA/softcite_dataset")
    document_level_parser.add_argument("-p", "--prediction_field",
                                       help="Field name with results in the results dataset.", default="software")
    document_level_parser.add_argument("-g", "--gt_field", help="Field name with ground truth in the gold dataset.",
                                       default="software")
    document_level_parser.add_argument("-s", "--split", help="Split of the gold dataset.", default="test")
    document_level_parser.add_argument("-c", "--config", help="Config name of the gold dataset.", default="documents")
    document_level_parser.add_argument("-i", "--id",
                                       help="Name of field with unique identifier in the gold dataset. Is used to make sure that the results and gold are aligned.",
                                       default="id")
    document_level_parser.add_argument("-r", "--results_id",
                                       help="Name of field with unique identifier in the results dataset. Is used to make sure that the results and gold are aligned.",
                                       default="id")
    document_level_parser.add_argument("--hf_cache", help="Path to the Hugging Face cache.", default=None)
    document_level_parser.add_argument("--allow_subset", help="Allow evaluation of subset of the gold dataset.",
                                       action="store_true")
    document_level_parser.add_argument("--normalize_properties", help="Normalize properties.", action="store_true")
    document_level_parser.add_argument("--results_mapping",
                                       help="Mapping used to transform software attributes (json file). Could be used to convert labels from different datasets to the same format.",
                                       default=None)
    document_level_parser.add_argument("--gold_mapping",
                                       help="Mapping used to transform software attributes (json file). Could be used to convert labels from different datasets to the same format.",
                                       default=None)
    document_level_parser.add_argument("--only_with_mention",
                                       help="Evaluates only samples with at least one mention. In prediction or ground truth.",
                                       action="store_true")
    document_level_parser.add_argument("--only_with_annotation",
                                       help="Evaluates only samples with at least one mention in ground truth.",
                                       action="store_true")
    document_level_parser.set_defaults(func=document_level)

    intent_parser = subparsers.add_parser("intent", help="Citation intent classification evaluation.")
    intent_parser.add_argument("results", help="Path to the results file.")
    intent_parser.add_argument("--gold",
                               help="Name/path of the gold Hugging Face dataset.",
                               default="SoFairOA/software_intent_softcite_somesci_czi")
    intent_parser.add_argument("-p", "--prediction_field",
                               help="Field name with results in the results dataset.", default="label")
    intent_parser.add_argument("-g", "--gt_field", help="Field name with ground truth in the gold dataset.",
                               default="label")
    intent_parser.add_argument("-s", "--split", help="Split of the gold dataset.", default="test")
    intent_parser.add_argument("-c", "--config", help="Config name of the gold dataset.", default="default")
    intent_parser.add_argument("-i", "--id",
                               help="Name of field with unique identifier in the gold dataset. Is used to make sure that the results and gold are aligned.",
                               default="id")
    intent_parser.add_argument("-r", "--results_id",
                               help="Name of field with unique identifier in the results dataset. Is used to make sure that the results and gold are aligned.",
                               default="id")
    intent_parser.add_argument("--hf_cache", help="Path to the Hugging Face cache.", default=None)
    intent_parser.add_argument("--allow_subset", help="Allow evaluation of subset of the gold dataset.",
                               action="store_true")
    intent_parser.set_defaults(func=intent)

    args = parser.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == "__main__":
    main()
