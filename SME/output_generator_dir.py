import json
import os
from datetime import datetime

def generate_output(all_entities, output_dir, start_time, end_time, args):
    def get_incremental_filename(output_dir, base_name="runtime_results", extension="jsonl"):
        i = 1
        while os.path.exists(os.path.join(output_dir, f"{base_name}_{i}.{extension}")):
            i += 1
        return os.path.join(output_dir, f"{base_name}_{i}.{extension}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata = {
            "Start_Time": start_time.isoformat(),
            "End_Time": end_time.isoformat(),
            "LLM_Model": args.model,
            "Temperature": args.temperature,
            "Split_Type": args.split_type,
            "Window_Size": args.window_size,
            "Overlap_Sentences": args.overlap_sentences,
            "Batch_Processing": args.batch_processing
        }

    output_filename = get_incremental_filename(output_dir)
    with open(output_filename, 'w', encoding='utf-8') as file:
            for document in all_entities:
                document_id = document["document_id"]
                entities = document["softwares"]
                if not entities:
                    full_entity = {
                        "id": document_id,
                        "software": [],
                        "version": [],
                        "publisher": [],
                        "url": [],
                        "language": [],
                        **metadata
                    }
                    json_line = json.dumps(full_entity, ensure_ascii=False)
                    file.write(f"{json_line}\n")
                else:
                    for entity in entities:
                        full_entity = {"id": document_id, **entity, **metadata}
                        json_line = json.dumps(full_entity, ensure_ascii=False)
                        file.write(f"{json_line}\n")

    print(f"Output saved in: {output_filename}")