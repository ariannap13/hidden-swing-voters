import pandas as pd
import numpy as np
import json
from pt_model import instantiate_model, predict, predict_raw, read_config
from pt_postprocessing import aggregate_results, output_to_json, agg_results_segments_row, process_labels_map, PersuationResults, postprocess_scores
import pt_constants

import torch 

# check general information
print('Pytorch version: ', torch.__version__)
print('Check availability: ', torch.cuda.is_available())

# check if we have cuda installed
if torch.cuda.is_available():
    # to use GPU
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('GPU is:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu") 