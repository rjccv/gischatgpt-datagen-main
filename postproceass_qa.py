import sqlite3
import json
import time  # Import the time module
import re


from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ProcessQAGen:
    db_file: str = field(metadata={'help': 'Path to the .db file'})
    output_json_file: str = field(default="M7_im2gps3k_gis_instruction_set.json", metadata={'help': 'Output JSON file name'})


def parse_qa(text):
    # Regular expressions for questions and answers
    question_pattern = re.compile(r'(Q\d*:|Question:|\*\*Q:\*\*|\*\*Q \d{1,2}:\*\*|\*\*Question \d{1,2}:\*\*)')
    answer_pattern = re.compile(r'(A\d*:|Answer:|\*\*A:\*\*|\*\*Answer:\*\*|\*\*Answer \d{1,2}:\*\*)')

    # Patterns to remove from the start of questions and answers
    remove_q_pattern = re.compile(r'^Q\d*:|^Question:\s*|^\*\*Q:\*\*\s*|^\*\*Q \d{1,2}:\*\*\s*|^\*\*Question \d{1,2}:\*\*\s*')
    remove_a_pattern = re.compile(r'^A\d*:|^Answer:\s*|^\*\*A:\*\*\s*|^\*\*Answer:\*\*\s*|^\*\*Answer \d{1,2}:\*\*\s*')

    questions_answers = {}
    current_question = None

    for line in text.split('\n'):
        if question_pattern.match(line):
            # Remove the question prefix
            question = remove_q_pattern.sub('', line).strip()
            current_question = question
        elif answer_pattern.match(line) and current_question:
            # Remove the answer prefix
            answer = remove_a_pattern.sub('', line).strip()
            questions_answers[current_question] = answer
            current_question = None

    return questions_answers


def read_db_and_process(ARGS):
    # Connect to the SQLite database
    conn = sqlite3.connect(ARGS.db_file)
    cursor = conn.cursor()

    # Query to select img and conversation columns
    cursor.execute("SELECT id, img_id, question, lat, lon FROM data")
    # cursor.execute("SELECT id, img_id, question FROM data")

    list_of_instruction_tuning = []
    # Process each record
    for i, (id, img_id, conversation, lat, lon) in enumerate(cursor):
    # for i, (id, img_id, conversation) in enumerate(cursor):
        qa_pairs = parse_qa(conversation)

        # Prepare the output content
        for j, (question, answer) in enumerate(qa_pairs.items()):
            output_content = {'id': img_id, 'img': f'{img_id}.pkl', 'conversations': []}
            if j % 2 == 0:
                output_content['conversations'].append({'from': 'human', 'value': f"{question}\n<image>"})
            else:
                output_content['conversations'].append({'from': 'human', 'value': f"<image>\n{question}"})
            output_content['conversations'].append({'from': 'gpt', 'value': answer})
            output_content['lat'] = lat
            output_content['long'] = lon
            list_of_instruction_tuning.append(output_content)
    
    # Write to a JSON file
    with open(ARGS.output_json_file, 'w') as json_file:
        json.dump(list_of_instruction_tuning, json_file, indent=4)

    # Close the database connection
    conn.close()



def main():
    parser = HfArgumentParser((ProcessQAGen))
    process_qa_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    read_db_and_process(process_qa_args)

if __name__ == "__main__":
    main()
