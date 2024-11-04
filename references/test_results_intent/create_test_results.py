import json
import random
from pathlib import Path

from datasets import load_dataset

SCRIPT_PATH = Path(__file__).parent
dataset = load_dataset("SoFairOA/software_intent_softcite_somesci_czi")

with (open(SCRIPT_PATH / "all_ok.jsonl", "w") as all_ok_f, open(SCRIPT_PATH / "all_bad.jsonl", "w") as all_bad_f,
      open(SCRIPT_PATH / "random.jsonl", "w") as random_f):
    for s in dataset["test"]:

        print(json.dumps({
            "id": s["id"],
            "label": s["label"]
        }), file=all_ok_f)

        print(json.dumps({
            "id": s["id"],
            "label": (s["label"] + 1) % 4
        }), file=all_bad_f)


        print(json.dumps({
            "id": s["id"],
            "label": random.randint(0, 3)
        }), file=random_f)
