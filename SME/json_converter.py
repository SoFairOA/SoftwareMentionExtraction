import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Union, Optional

class JSONLProcessor:
    """
    A class for processing JSONL files, grouping data by ID, deduplicating software entries,
    and generating output in a required format.
    """
    
    def __init__(self, results_path: str, output_path: str):
        """
        Initialize the JSONLProcessor with file paths.

        :param results_path: Path to the JSONL file containing the model results.
        :param output_path: Path for the output JSONL file.
        """
        
        self.results_path: str = results_path
        self.output_path: str = output_path
        
    def load_results(self) -> List[Dict[str, Union[str, List[Dict[str, Optional[Union[str, List[str]]]]]]]]:
        """
        Load the JSONL file containing the model predictions.
        
        :return: A list of dictionaries containing the results.
        """
        
        results: List[Dict[str, Union[str, List[Dict[str, Optional[Union[str, List[str]]]]]]]] = []
        with open(self.results_path, "r", encoding="utf-8") as file:
            for line in file:
                results.append(json.loads(line.strip()))  
        return results
    
    
    @staticmethod
    def group_and_deduplicate_software_by_id(
        results: List[Dict[str, Union[str, List[Dict[str, Optional[Union[str, List[str]]]]]]]]
    ) -> List[Dict[str, Union[str, List[Dict[str, Optional[Union[str, List[str]]]]]]]]:
        import re
        import unicodedata
        import copy

        def normalize_string(s: str) -> str:
            """
            Normalize a string by converting it to lowercase, removing punctuation, and extra spaces.
            """
            s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('ASCII')
            s = s.lower()
            s = re.sub(r'[^\w\s]', '', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        grouped_data = defaultdict(lambda: {'id': None, 'software': []})

        placeholders = {
            "<no URL mentioned>",
            "<no URL specified>",
            "(no URL mentioned)",
            "<no language specified>",
            "(no language specified)",
            "no language specified",
            "<no programming language mentioned>",
            "(no programming language mentioned)",
            "no programming language mentioned",
            "<no version specified>",
            "(no version specified)",
            "no version specified",
            "(no version mentioned)",
            "<no publisher specified>",
            "(no publisher specified)",
            "no publisher specified",
            "(no publisher mentioned)",
            "unspecified",
            "<unspecified>",
            "(unspecified)",
            "undefined",
            "<undefined>",
            "(undefined)",
            "no version mentioned",
            "<no version mentioned>",
            "(no version mentioned)",
            "no publisher mentioned",
            "<no publisher mentioned>",
            "(no publisher mentioned)",
            "None",
            "None mentioned",
            "N/A",
            "Ver. ",
            "-",
            ""
        }

        normalized_placeholders = set(normalize_string(p) for p in placeholders)

        def clean_version(version_list):
            cleaned_versions = []
            for version in version_list:
                if isinstance(version, str):
                    cleaned_version = version.replace("Ver. ", "").replace("Version ", "").strip()
                    if cleaned_version and normalize_string(cleaned_version) not in normalized_placeholders:
                        cleaned_versions.append(cleaned_version)
                else:
                    pass
            return cleaned_versions

        def flatten_and_clean_field(field_list):
            cleaned_list = []
            for item in field_list:
                if isinstance(item, list):
                    cleaned_list.extend(flatten_and_clean_field(item))
                elif isinstance(item, str):
                    item_cleaned = item.strip()
                    if item_cleaned and normalize_string(item_cleaned) not in normalized_placeholders:
                        cleaned_list.append(item_cleaned)
                else:
                    pass
            return cleaned_list

        for result in results:
            id: Optional[str] = result.get("id")
            if not id:
                id = f"generated_id_{len(grouped_data)}"
            
            software_list = result.get("software", [])
            if software_list is None:
                continue

            if isinstance(software_list, str):
                software_list = [software_list]

            for software in software_list:
                if isinstance(software, dict):
                    software = copy.deepcopy(software)
                    if 'name' in software and isinstance(software['name'], str):
                        software['name'] = software['name'].strip()
                        name_normalized = normalize_string(software['name'])
                        if not software['name'] or name_normalized in normalized_placeholders:
                            continue 
                    else:
                        continue  

                    for field in ["version", "publisher", "url", "language"]:
                        if field in software:
                            if isinstance(software[field], str):
                                software[field] = [software[field]]
                            elif not isinstance(software[field], list):
                                software[field] = []
                            software[field] = flatten_and_clean_field(software[field])
                            if field == "version":
                                software[field] = clean_version(software[field])
                        else:
                            software[field] = []
                    grouped_data[id]['software'].append(software)
                elif isinstance(software, str):
                    software_entry = {
                        "name": software.strip(),
                        "version": result.get("version", []),
                        "publisher": result.get("publisher", []),
                        "url": result.get("url", []),
                        "language": result.get("language", [])
                    }

                    name_normalized = normalize_string(software_entry['name'])
                    if not software_entry['name'] or name_normalized in normalized_placeholders:
                        continue  

                    for field in ["version", "publisher", "url", "language"]:
                        if isinstance(software_entry[field], str):
                            software_entry[field] = [software_entry[field]]
                        elif not isinstance(software_entry[field], list):
                            software_entry[field] = []
                        software_entry[field] = flatten_and_clean_field(software_entry[field])
                        if field == "version":
                            software_entry[field] = clean_version(software_entry[field])

                    grouped_data[id]['software'].append(software_entry)
                else:
                    pass
                
            grouped_data[id]['id'] = id

        deduplicated_grouped_data = []
        for doc_id, doc_content in grouped_data.items():
            seen_keys = set()
            deduped_software = []
            for software in doc_content['software']:
                name = software.get('name', '')
                if not name:
                    continue 
                name_normalized = normalize_string(name)
                version_list = software.get('version', [])
                version_normalized = ','.join([normalize_string(v) for v in version_list]) if version_list else ''
                key = (name_normalized, version_normalized)
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduped_software.append(software)
                else:
                    print(f"Duplicate found in document {doc_id}: {software}")
            deduplicated_document = {
                "id": doc_id,
                "software": deduped_software
            }
            deduplicated_grouped_data.append(deduplicated_document)

        return deduplicated_grouped_data



    def generate_output(self, results: List[Dict[str, Union[str, List[Dict[str, Optional[Union[str, List[str]]]]]]]]) -> None:
        """
        Generate the output JSONL file in the required format (deduplicated).
        
        :param results: The prediction results.
        """
        grouped_and_deduplicated_results = self.group_and_deduplicate_software_by_id(results)
        with open(self.output_path, "w", encoding="utf-8") as f:
            for item in grouped_and_deduplicated_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")  

    def process(self) -> None:
        """
        Main processing logic: load results, group and deduplicate them, and save the output.
        """
        results = self.load_results()
        self.generate_output(results)
        print(f"Processed and deduplicated results saved to {self.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSONL file, deduplicate software entries, and generate grouped output.")
    parser.add_argument("results", help="Path to the JSONL file containing the model results.")
    parser.add_argument("output", help="Path for the output JSONL file.")
    args = parser.parse_args()

    processor = JSONLProcessor(args.results, args.output)
    processor.process()
