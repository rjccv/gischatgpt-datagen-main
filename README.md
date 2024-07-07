# GIS-ChatGPT Datagen

## Description

This project aims to generate a question-answer dataset based on reformatted OpenStreetMap (OSM) data using advanced natural language processing models. It involves two primary scripts: `gen_qa_llama_hf.py` for processing the OSM data and generating question-answer pairs, and `postprocess.py` for further processing and organizing these pairs.

## Getting Started
## Create Conda Env
```bash
conda create -n gisdatagen
conda activate gisdatagen
```

### Dependencies

- Python 3.11
- Standard Python Libraries: `pdb`, `re`, `os`, `sqlite3`, `json`, `time`
- Third-party Libraries:
  - `tqdm`
  - `transformers` (from Hugging Face)
  - `torch` (PyTorch)
  - `tiktoken` (Add description or installation instructions for this library)

Install the necessary libraries using the following command:

```bash
pip install tqdm transformers torch # Add any other libraries as needed
```

## Installing
Clone this repository to your local machine.

## Collect OSM Metadata
Run extract_osm.py
This script makes api calls to OSM servers and collects the metadata.

## Executing the Program
Running gen_qa_llama_hf.py
This script processes the OSM data and generates question-answer pairs. To run this script, you need to provide certain parameters, such as the path to the OSM JSON file and Hugging Face model details.

Example command:
```bash
python gen_qa_llama_hf.py --osm_json_file "path/to/osm_data.json" --hf_token "your_huggingface_token" --model_id "model_identifier"
```

Replace "path/to/osm_data.json", "your_huggingface_token", and "model_identifier" with your actual file path, Hugging Face token, and model ID, respectively.

Running postprocess.py
After generating the QA pairs, use the postprocess.py script to process and organize these pairs. This script also requires the path to the database file created by the first script and an output JSON file name.

Example command:
```bash
python postprocess.py --db_file "path/to/database.db" --output_json_file "output_file_name.json"
```

Replace "path/to/database.db" and "output_file_name.json" with your actual database file path and desired output JSON file name.

## Help
If you encounter any issues or have questions, please feel free to open an issue in this repository.

## Authors
Brian Dina
brian.n.dina@gmail.com

Version History
0.1
Initial Release


## License
This project is licensed under the [LICENSE NAME] License - see the LICENSE.md file for details