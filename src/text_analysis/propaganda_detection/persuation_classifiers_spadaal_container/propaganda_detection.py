import os
import json
import torch
import warnings

from pt_model import instantiate_model, predict, read_config
from pt_postprocessing import aggregate_results, output_to_json
import time

# Ignoring warnings
warnings.filterwarnings('ignore')

# start timer
start = time.time()

# Check for CUDA availability and device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda_status = "available" if torch.cuda.is_available() else "not available"
print(f'PyTorch version: {torch.__version__}, CUDA is {cuda_status}.', flush=True)
if torch.cuda.is_available():
    print(f'There are {torch.cuda.device_count()} GPU(s) available: {torch.cuda.get_device_name(0)}', flush=True)

# Loading data function to encapsulate JSON file operations
def load_data(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

# Model configuration and initialization
cfg = read_config('defaults.cfg')
model = instantiate_model(cfg)

# Data loading and processing
data = load_data("../tweets_vips.json")
print(f"Loaded {len(data)} tweets.", flush=True)
# subset data to only first 10 tweets for testing
#data = {k: data[k] for k in list(data.keys())[:10]}
guid = list(data.keys())
text = [data[id]['text'] for id in guid]

# Execute prediction and aggregation
output = predict(model, text)
results_sentence = aggregate_results(text, output, level='sentence')
results_paragraph = aggregate_results(text, output, level='paragraph', granularity='fine', detailed_results=True)

# Streamlining the annotation snippet function
def get_annotated_snippets(results_sentence, guid, text, data):
    json_output = output_to_json(results_sentence, document_ids=guid, map_to_labels=True)

    for doc_id, doc in json_output.items():
        idx = guid.index(doc_id)
        sent = text[idx]
        for sentence in doc:
            sentence_chunk = sent[sentence['start']:sentence['end']]
            if 'annotations' not in data[doc_id]:  # Initialize annotations list if not present
                data[doc_id]['annotations'] = []
            data[doc_id]['annotations'].append({'label': sentence['label'], 'sentence_chunk': sentence_chunk})
    return data

data_annotated = get_annotated_snippets(results_sentence, guid, text, data)

# Encapsulate file saving functionality
def save_data(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

save_data("../tweets_vips_annotated.json", data_annotated)

# end timer
end = time.time()
print(f"Execution time: {end - start} seconds", flush=True)
