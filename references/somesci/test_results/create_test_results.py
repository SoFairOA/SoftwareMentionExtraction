import json
import csv
from pathlib import Path

from datasets import load_dataset

SCRIPT_PATH = Path(__file__).parent
dataset = load_dataset("SoFairOA/somesci_dataset")


with (open(SCRIPT_PATH / "all_ok.jsonl", "w") as all_ok_f, open(SCRIPT_PATH / "empty.jsonl", "w") as empty_f,
      open(SCRIPT_PATH / "first.jsonl", "w") as first_f, open(SCRIPT_PATH / "all_ok.csv", "w") as all_ok_csv_f,
      open(SCRIPT_PATH / "all_ok_subset.csv", "w") as all_ok_subset_f):
    writer = csv.DictWriter(all_ok_csv_f, fieldnames=["id", "labels", "software"])
    writer.writeheader()
    writer_subset = csv.DictWriter(all_ok_subset_f, fieldnames=["id", "labels", "software"])
    writer_subset.writeheader()
    for i, s in enumerate(dataset["test"]):
        # convert dict of lists to list of dicts
        software = [{k: v[i] for k, v in s["software"].items()} for i in range(len(s["software"]["name"]))]
        writer.writerow({
            "id": s["id"],
            "labels": json.dumps(s["labels"]),
            "software": json.dumps(software),
        })
        if i < 10:
            writer_subset.writerow({
                "id": s["id"],
                "labels": json.dumps(s["labels"]),
                "software": json.dumps(software),
            })

        print(json.dumps({
            "id": s["id"],
            "labels": s["labels"],
            "software": software,
        }), file=all_ok_f)

        print(json.dumps({
            "id": s["id"],
            "labels": [0] * len(s["labels"]),
            "software": [],
        }), file=empty_f)

        labels = [0] * len(s["labels"])
        try:
            first_label_index = s["labels"].index(1)
            while s["labels"][first_label_index] == 1 or s["labels"][first_label_index] == 2:
                labels[first_label_index] = s["labels"][first_label_index]
                first_label_index += 1

        except ValueError:
            pass

        print(json.dumps({
            "id": s["id"],
            "labels": labels,
            "software": [software[0]] if len(software) > 0 else [],
        }), file=first_f)
