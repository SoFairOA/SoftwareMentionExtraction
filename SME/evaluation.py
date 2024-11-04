import copy
import json
from argparse import ArgumentParser

import evaluate
import datasets
from datasets import load_dataset


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


def load_and_check_datasets(results_path: str, gold_path: str, results_id: str, gold_id: str, results_field: str, gold_field: str, split: str = "test", config: str = None, hf_cache: str = None):
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
    :return: Gold and results datasets.
    """

    # load

    results_dataset = load_dataset("json", data_files=results_path)["train"]
    gold_dataset = load_dataset(gold_path, split=split, name=config, cache_dir=hf_cache)

    results_dataset = results_dataset.select_columns([results_id, results_field])
    gold_dataset = gold_dataset.select_columns([gold_id, gold_field])

    # check
    results_ids = set(results_dataset[results_id])
    gold_ids = set(gold_dataset[gold_id])

    if results_ids != gold_ids:
        print("Results and gold are not aligned. Exiting.")
        exit(1)

    return gold_dataset, results_dataset


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
        gold.append(gold_example[gold_field])
        results.append(results_mapping[gold_example[gold_id]])

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


def sequence_labeling(args):
    """
    Evaluation of sequence labeling.

    :param args: User arguments.
    """

    # load
    gold_dataset, results_dataset = load_and_check_datasets(args.results, args.gold, args.results_id, args.id, args.prediction_field, args.gt_field, args.split, args.config, args.hf_cache)

    # translate the labels
    if not args.disable_translate and (hasattr(gold_dataset.features[args.gt_field], "feature") and hasattr(gold_dataset.features[args.gt_field].feature, "names")):
        results_dataset = results_dataset.map(
            LabelsTranslator(args.prediction_field, gold_dataset.features[args.gt_field].feature.names),
            load_from_cache_file=False,
            keep_in_memory=True
        )
        # for some reason it was necessary to specify this casting
        gold_features = gold_dataset.features.copy()
        gold_features[args.gt_field] = datasets.Sequence(datasets.Value("string"))

        gold_dataset = gold_dataset.map(
            LabelsTranslator(args.gt_field, gold_dataset.features[args.gt_field].feature.names),
            load_from_cache_file=False,
            keep_in_memory=True,
            features=gold_features
        )

    # align the results and gold
    gold, results = align_datasets(gold_dataset, results_dataset, args.results_id, args.id, args.prediction_field, args.gt_field)

    # evaluate
    seqeval = evaluate.load("seqeval")
    eval_res = seqeval.compute(predictions=results, references=gold)
    # convert all numpy types to python types, so we can serialize it to json
    eval_res = convert_numpy_types(eval_res)
    print(json.dumps(eval_res))


def document_level(args):
    """
    Document level evaluation.
    :param args: User arguments.
    """

    # load
    gold_dataset, results_dataset = load_and_check_datasets(args.results, args.gold, args.results_id, args.id, args.prediction_field, args.gt_field, args.split, args.config, args.hf_cache)

    # align the results and gold
    gold, results = align_datasets(gold_dataset, results_dataset, args.results_id, args.id, args.prediction_field, args.gt_field)

    # parse software names
    gold_software_names, results_software_names = [], []

    gold_properties = {
        "version": [],
        "publisher": [],
        "url": [],
        "language": []
    }
    results_properties = copy.deepcopy(gold_properties)
    gold_properties_independent, results_properties_independent = copy.deepcopy(gold_properties), copy.deepcopy(gold_properties)

    for i in range(len(gold)):
        gold_software_names.append(gold[i]["name"])
        results_software_names.append([x["name"] for x in results[i]])

        for prop in gold_properties:
            gold_properties[prop].append([
                f"{s_name} {p}" for i_s, s_name in enumerate(gold[i]["name"]) for p in gold[i][prop][i_s]
            ])
            results_properties[prop].append([
                f"{soft['name']} {p}" for i_s, soft in enumerate(results[i]) for p in soft[prop]
            ])

            gold_properties_independent[prop].append([
                p for i_s, s_name in enumerate(gold[i]["name"]) for p in gold[i][prop][i_s]
            ])

            results_properties_independent[prop].append([
                p for i_s, soft in enumerate(results[i]) for p in soft[prop]
            ])

    # evaluate
    doc_level = evaluate.load("mdocekal/multi_label_precision_recall_accuracy_fscore")
    doc_level.info.features = datasets.Features({   # we need to specify this, because the HF evaluator is not able to infer data types correctly when the first example is empty
        'predictions': datasets.Sequence(datasets.Value('string')),
        'references': datasets.Sequence(datasets.Value('string')),
    })
    eval_res = doc_level.compute(predictions=results_software_names, references=gold_software_names)
    print("Document level software mentions extraction evaluation:")
    print(json.dumps(eval_res))

    # evaluate properties
    for prop in gold_properties:
        eval_res = doc_level.compute(predictions=results_properties[prop], references=gold_properties[prop])
        print(f"Document level {prop} extraction evaluation:")
        print("\t" + json.dumps(eval_res))
        print("\tIndependent evaluation:")
        eval_res = doc_level.compute(predictions=results_properties_independent[prop], references=gold_properties_independent[prop])
        print("\t\t" + json.dumps(eval_res))


def intent(args):
    """
    Evaluation of intent classification.

    :param args: User arguments.
    """
    # load
    gold_dataset, results_dataset = load_and_check_datasets(args.results, args.gold, args.results_id, args.id, args.prediction_field, args.gt_field, args.split, args.config, args.hf_cache)

    # align the results and gold
    gold, results = align_datasets(gold_dataset, results_dataset, args.results_id, args.id, args.prediction_field, args.gt_field)

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

    parser = ArgumentParser(description="Script for evaluation of software mentions extraction. The results are printed to stdout.")
    subparsers = parser.add_subparsers()

    sequence_labeling_parser = subparsers.add_parser("sequence_labeling", help="Sequence labeling evaluation.")
    sequence_labeling_parser.add_argument("results", help="Path to the results file.")
    sequence_labeling_parser.add_argument("--gold",
                                          help="Name/path of the gold Hugging Face dataset.",
                                          default="SoFairOA/softcite_dataset")
    sequence_labeling_parser.add_argument("-p", "--prediction_field", help="Field name with results in the results dataset.", default="labels")
    sequence_labeling_parser.add_argument("-g", "--gt_field", help="Field name with ground truth in the gold dataset.", default="labels")
    sequence_labeling_parser.add_argument("-s", "--split", help="Split of the gold dataset.", default="test")
    sequence_labeling_parser.add_argument("-c", "--config", help="Config name of the gold dataset.", default="documents")
    sequence_labeling_parser.add_argument("-i", "--id", help="Name of field with unique identifier in the gold dataset. Is used to make sure that the results and gold are aligned.", default="id")
    sequence_labeling_parser.add_argument("-r", "--results_id", help="Name of field with unique identifier in the results dataset. Is used to make sure that the results and gold are aligned.", default="id")
    sequence_labeling_parser.add_argument("--hf_cache", help="Path to the Hugging Face cache.", default=None)
    sequence_labeling_parser.add_argument("--disable_translate", help="Disables translation of the results and gold.", action="store_true")
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
    intent_parser.set_defaults(func=intent)

    args = parser.parse_args()

    if args is not None:
        args.func(args)
    else:
        exit(1)


if __name__ == "__main__":
    main()
