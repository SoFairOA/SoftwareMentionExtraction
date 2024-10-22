import re
import csv
import os

def generate_output(softwares, start_time, end_time, prompt_used, document_type, args, output_dir, top_p, top_k, max_tokens, window_size, overlap_sentences, document_id):
    #-----A SEQUENCE OF CLEANING FUNCTIONS TO APPLY ON THE OUTPUT OF THE MODEL AND ON THE FINAL CSV OUTPUT-----#
    
    def normalize_software_name(software_name):
        return software_name.strip('→"').strip()

    def filter_non_software_mentions(software_list):
        pattern = re.compile(r'(None|"None"|→|"→"|\s*There are no software names\s*|\s*There are no explicit software names\s*)', re.IGNORECASE)
        filtered = []
        for software in software_list:
            normalized_software = normalize_software_name(software)
            if not pattern.match(normalized_software):
                filtered.append(normalized_software)
        return filtered
    #------GENERATE AN INCREMENTAL NAME FOR GENERATED OUTPUT TO KEEP TRACK OF EACH RUN------#
    def get_incremental_filename(output_dir, base_name="runtime_results", extension="csv"):
        i = 1
        while os.path.exists(os.path.join(output_dir, f"{base_name}_{i}.{extension}")):
            i += 1
        return os.path.join(output_dir, f"{base_name}_{i}.{extension}")

    #-----A REGEX-BASED METHOD TO FIND THE VERSION NUMBER OF EXTRACTED SOFTWARE-----#
    def extract_version(software_name):
        version_pattern = r'(?:v(?:er(?:sion)?)?\.?\s*(\d+(\.\d+)*))|(\d+(\.\d+)*)'
        match = re.search(version_pattern, software_name, re.IGNORECASE)
        
        if match:
            #-----GROUP 1 WILL CONTAIN SOFTWARE WITH "VER", "VERSION" OR "V." IN THEIR NAME
            version = match.group(1) if match.group(1) else match.group(0)
            #-----REMOVE THE VERSION FROM THE SOFTWARE NAME-----#
            software_name_without_version = re.sub(version_pattern, '', software_name, flags=re.IGNORECASE).strip()
            software_name_without_version = re.sub(r'[-_.]+$', '', software_name_without_version)
            software_name_without_version = re.sub(r'^[-_.]+', '', software_name_without_version)
            return version, software_name_without_version
        else:
            return "N/A", software_name


    #-----CREATE OUTPUT DIRECTORY IF IT DOESN'T EXITS-----#
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory di output creata: {output_dir}")


    filtered_softwares = filter_non_software_mentions(softwares)
    output_filename = get_incremental_filename(output_dir)

    #-----WRITE RESULTS ON A CSV-----#
    with open(output_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Document_ID', 'Software_Name', 'Version', 'Action', 'Position', 'Start_Time', 'End_Time', 'LLM_Model', 
            'Temperature', 'Split_Type', 'Document_Type', 'Prompt_Type', 'Top_P', 
            'Top_K', 'Max_Tokens', 'Window_Size', 'Overlap_Sentences'
        ])

        #-----ITERATES ON THE REMAINING MENTIONS, THEN SPLITS THEM BASED ON THE "|" CHARACTER,-----#
        #-----WHICH THE MODEL USED, AS INSTRUCTED, AS A SEPARATING CHARACTER FOR THE ACTION-----#
        #-----("USED", "SHARED", "CREATED") AND THE POSITION OF THE SOFTWARE MENTION EXTRACTED-----#
        for software in filtered_softwares:
            software_parts = software.split('|')
            if len(software_parts) == 3:
                software_name, action, position = software_parts
            else:
                software_name, action, position = software, "unknown", "unknown"
            version_number, software_name_without_version = extract_version(software_name)
            writer.writerow([
                document_id, software_name_without_version, version_number, action, position, start_time, end_time, args.model, 
                args.temperature, args.split_type, document_type, args.prompt_type, 
                top_p, top_k, max_tokens, window_size, overlap_sentences
            ])
    print(f"Output salvato in: {output_filename}")
