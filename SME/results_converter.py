import pandas as pd
import re
from typing import Optional
import argparse

class TxtToExcelConverter:
    """
    A class to convert structured metrics from a .txt file to an Excel file.
    """

    def __init__(self, input_file_path: str, output_file_path: str):
        """
        Initializes the converter with file paths.

        :param input_file_path: Path to the input .txt file containing structured metrics.
        :param output_file_path: Path to save the output Excel file.
        """
        self.input_file_path: str = input_file_path
        self.output_file_path: str = output_file_path

    def convert_to_excel(self) -> None:
        """
        Parses the input file and converts it to an Excel file.
        """
        categories: list[str] = []
        precisions: list[Optional[float]] = []
        recalls: list[Optional[float]] = []
        accuracies: list[Optional[float]] = []
        fscores: list[Optional[float]] = []

        category_pattern = re.compile(r"^(.* evaluation):")
        metric_pattern = re.compile(r"\{.*?\}")

        with open(self.input_file_path, 'r') as file:
            current_category: Optional[str] = None
            for line in file:
                line = line.strip()
                category_match = category_pattern.match(line)

                if category_match:
                    current_category = category_match.group(1)

                elif metric_pattern.match(line):
                    metrics = eval(line)  
                    categories.append(current_category)
                    precisions.append(metrics.get("precision"))
                    recalls.append(metrics.get("recall"))
                    accuracies.append(metrics.get("accuracy"))
                    fscores.append(metrics.get("fscore"))

        df = pd.DataFrame({
            "Category": categories,
            "Precision": precisions,
            "Recall": recalls,
            "Accuracy": accuracies,
            "F-Score": fscores
        })

        df.to_excel(self.output_file_path, index=False)
        print(f"Data saved to {self.output_file_path}")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert structured metrics from a .txt file to an Excel file.")
    parser.add_argument("input_file", type=str, help="Path to the input .txt file.")
    parser.add_argument("output_file", type=str, help="Path to where to save the output Excel file.")
    
    args = parser.parse_args()
    
    converter = TxtToExcelConverter(args.input_file, args.output_file)
    converter.convert_to_excel()