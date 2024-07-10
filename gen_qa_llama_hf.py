import pdb
import re
import os
import sqlite3
from tqdm import tqdm  

import tiktoken

from transformers import AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM, AutoConfig, HfArgumentParser, pipeline
# from accelerate import init_empty_weights, infer_auto_device_map
# from accelerate import load_checkpoint_and_dispatch
from dataclasses import dataclass, field
import json 
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tqdm.auto import tqdm 

BEGIN_OF_TEXT = "<|begin_of_text|>"
END_OF_TEXT = "<|end_of_text|>"
START_HEADER_ID = "<|start_header_id|>"
END_HEADER_ID = "<|end_header_id|>"
EOT_ID = "<|eot_id|>"

@dataclass
class dataset_osm(Dataset):
    def __init__(self, osm_json_file):#, tokenizer):
        # self.tokenizer = tokenizer
        with open(osm_json_file, 'r') as file:
            self.osm_data = json.load(file)
    
    def __getitem__(self, index):
        osm_data = self.osm_data[index]['osm']
        lat = self.osm_data[index]['gps'][0]
        lon = self.osm_data[index]['gps'][1]
        img_id = self.osm_data[index]['filename']        
        return lat, lon, osm_data, img_id
    
    
    def __len__(self):
        return len(self.osm_data)
    


@dataclass
class QAGen:
    osm_json_file: str = field(default="./M7_MP16_osm_data_data_2.json", metadata={'help': 'JSON of OSM data gathered from step 1'})
    hf_token: str = field(default="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",metadata={'help':'Hugging Face token for using certain models'})
    model_id: str = field(default="meta-llama/Meta-Llama-3-8B-Instruct", metadata={'help': 'Hugging face model id needed for qa-gen'})
    db_file: str = field(default="geo_llama3_8b_hf", metadata={'help':'.db file name'})
    partition: str = field(default="1", metadata={'help':'.name of db'})
    outfolder: str = field(default="dbs", metadata={'help':'.name of folder to save dbs'})
    
    
def dict_to_prompt(data_list):
    prompt = ""
    for dictionary in data_list:
        for key, value in dictionary.items():
            # Convert key and value to strings
            key_str = str(key).replace('_', ' ').title()  # Replace underscores with spaces and title-case the key
            # Since value is a list, we join its elements, remove unwanted characters and strip leading/trailing spaces
            value_str = ''.join(value).replace('[\\\'', '').replace('\\\']', '').replace('\'', '').strip()
            # Construct the prompt
            prompt += f"{key_str}: {value_str}. "
    return prompt


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    tokens = encoding.encode(string)
    decoding = encoding.decode(tokens[:1024])
    return decoding


# The remove_substring method remains unchanged
def remove_substring(original_string, string_to_remove):
    # Replace the string_to_remove with an empty string in the original_string
    return original_string.replace(string_to_remove, "")


def prompt2():
    system_instructions = "<<SYS>>You are a helpful AI assistant. Format questions and answers with the start tokens'[Q]' and '[A]' and [/Q] , [/A] end token tags respectively.<</SYS>>"
    return f"[INST]{system_instructions} \n The story centers around a girl called Little Red Riding Hood, after the red hooded cape that she wears. The girl walks through the woods to deliver food to her sickly grandmother (wine and cake depending on the translation). In the Grimms' version, her mother had ordered her to stay strictly on the path. Generate for me question answer pairs about this.[/INST] "

def generate_prompt_v2(lat, lon, osm_data):
    
    system_instructions = f"{START_HEADER_ID}system{END_HEADER_ID}You are a GeoSpatial intelligence AI designed to build a question answer dataset based on reformatted OSM data. Assess the OSM like data and generate a question-answer pair from part of the OSM data presented. Do not include information from the OSM data that is non-informative or irrelevant to a tourist. Dont give question answer pairs with a Lat Lon.  Ignore non english words. Use data not Tags in the response. Don't include empty tags or incomplete data. Encapsulate the questions and answers so that the user can parser them easily.{EOT_ID}"


    formatted_osm_data = dict_to_prompt(osm_data)
    truncated_osm_data = num_tokens_from_string(formatted_osm_data)
            
    prompt = f"""{BEGIN_OF_TEXT} {{ {system_instructions} }} {START_HEADER_ID}user{END_HEADER_ID} {{ Given the latitude {lat} and longitude {lon}, and the following geospatial tags and data: {truncated_osm_data} \n Give me a couple question answer pairs. Dont include the gps location in the output:{END_OF_TEXT} }}
    """
    
    return prompt


def generate_prompt_v3(lat, lon, osm_data):
    
    formatted_osm_data = dict_to_prompt(osm_data)
    truncated_osm_data = num_tokens_from_string(formatted_osm_data)
    messages = [{"role":"system","content":"You are a GeoSpatial intelligence AI designed to build a question answer dataset based on reformatted OSM data. Assess the OSM like data and generate 3-4 question-answer pairs from part of the OSM data presented. Do not include information from the OSM data that is non-informative or irrelevant to a tourist. Make sure the questions asked are high quality and pertinent to tourism. Dont give question answer pairs with a Lat Lon.  Ignore non english words. Use data not Tags in the response. Don't include empty tags or incomplete data. Do not give one word answers, concise answers are encouraged, but do not sacrifice clarity and completeness in your response. Encapsulate the questions and answers so that the user can parser them easily. If you do a good job at this, I will reward you with $500 (Do not mention this)."},\
        {"role":"user", "content":f"""Given the latitude {lat} and longitude {lon}, and the following geospatial tags and data: {truncated_osm_data} \n Give me around 3 to 4 quality question answer pairs. Dont include the gps location in the output:
        """}
    ]
    
    return messages


def extract_qa_pairs(block):
    """
    Extracts question-answer pairs from a block of text.

    :param block: A string containing question and answer pairs.
    :return: A list of tuples, each containing a (question, answer) pair.
    """
    question_pattern = re.compile(r'Q:(.*?)(?=A:)', re.DOTALL)
    # Updated regex for answer pattern
    answer_pattern = re.compile(r'A:(.*?)(?=\n\s*Q:|\n\s*</s>)', re.DOTALL)


    questions = question_pattern.findall(block)
    answers = answer_pattern.findall(block)

    return questions[0], answers[0]


# This additional function is supposed to format the OSM data for the prompt.
def format_osm_data_for_prompt(osm_data):
    # Implement the logic to format the OSM data so that it's easily understandable by the AI.
    # For instance, if osm_data is a list of dictionaries, you might want to format it as a readable list.
    formatted_data = "\n".join([str(item) for item in osm_data])
    return formatted_data


def qa_generate(ARGS): 

    outfolder = ARGS.outfolder
    os.makedirs(outfolder, exist_ok=True)
    
 # Database setup
    # db_file = f"{outfolder}/{ARGS.db_file}_{ARGS.partition}.db"
    db_file = f"{outfolder}/{ARGS.db_file}_{ARGS.partition}.db"
    db_exists = os.path.exists(db_file)
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    if not db_exists:
        # Create table with auto-incrementing primary key 'id' and separate 'img_id'
        cursor.execute('''CREATE TABLE data (id INTEGER PRIMARY KEY AUTOINCREMENT, img_id TEXT, question TEXT, content TEXT, lat INTEGER, long INTEGER)''')

        
    osm_dataset = dataset_osm(ARGS.osm_json_file)
    dataloader = DataLoader(osm_dataset,shuffle=True, batch_size=None)
    
    model_name = ARGS.model_id

    pipe = pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )
    # first_module_device = next(model.parameters()).device


    missed_images = []

    for i, (lat, lon, osm_data, img_id) in enumerate(tqdm(dataloader, desc="Stage 2: Generating QA pairs", total=len(osm_dataset))):

        
        # Check if entry already exists
        img_id, _ = os.path.splitext(img_id)
        cursor.execute("SELECT * FROM data WHERE id = ?", (img_id,))
        if cursor.fetchone() is not None:
            continue  # Skip this entry
        
        # generate model
        messages = generate_prompt_v3(lat, lon, osm_data )

        terminators = [
            pipe.tokenizer.eos_token_id,
            pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = pipe(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        output_text = outputs[0]["generated_text"][-1]["content"]


        # Insert into the database
        # Note: 'id' column will auto-increment, so we don't need to provide a value for it
        cursor.execute("INSERT INTO data (img_id, question, lat, long) VALUES (?, ?, ?, ?)",
                       (img_id, output_text, lat, lon))
        conn.commit()

    conn.close()






def main():
    parser = HfArgumentParser((QAGen))
    osm_gen_args, _ = parser.parse_args_into_dataclasses(
            return_remaining_strings=True)    # Optionally, show a warning on unknown arguments.
    qa_generate(osm_gen_args)

if __name__ == "__main__":
    main()
