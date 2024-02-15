import datetime
import json
from typing import List
from enum import Enum
from pydantic import BaseModel, Field

from pt_model import instantiate_model, predict, predict_raw, read_config
from pt_postprocessing import aggregate_results, output_to_json, agg_results_segments_row, process_labels_map, PersuationResults
import pt_constants
import torch


#text=["What is this but an EU kangaroo court where I am given 24 hours’ notice about allegations picked up from press stories, I did not receive any private money for political purposes, when representing Parliament in an official capacity, such gifts are reported in the register of gifts.","What is this but an EU kangaroo court where I am given 24 hours’ notice about allegations picked up from press stories, I did not receive any private money for political purposes, when representing Parliament in an official capacity, such gifts are reported in the register of gifts. If they try to bar me from the building, who else gives voice to the thousands of people who voted for me? Is this democracy EU style?"]

from fastapi import FastAPI, Response, Query, Form
from fastapi.responses import JSONResponse

app = FastAPI()

model_name = "pt"
version = "v1.0.0"

class DocBase(BaseModel):
    uid: str
    text: str

class PredictLevel(str, Enum):
    word = 'word'
    sentence = 'sentence'
    paragraph = 'paragraph'

class PredictReq(BaseModel):
    data: List[DocBase]
    map_to_labels: bool
    coarse_labels: bool
    level: PredictLevel # word, sentence, paragraph

class PredictDetailedReq(BaseModel):
    data: List[DocBase]
    coarse_labels: bool
    level: PredictLevel # word, sentence, paragraph
    
@app.get('/info')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


@app.get('/health')
async def service_health():
    """Return service health"""
    return {
        "state": "ok",
        "usingGPU": torch.cuda.is_available()
    }

@app.post('/predict')
async def model_predict(req: PredictReq):
    granularity = 'coarse' if req.coarse_labels else 'fine'
    text = [x.text for x in req.data]
    """Predict with input"""
    output =  predict(model, text, cfg=cfg)
    results_sentence = aggregate_results(text, output, level=req.level, granularity=granularity)
    return output_to_json(results_sentence , document_ids=[x.uid for x in req.data], map_to_labels=req.map_to_labels, coarse_labels=req.coarse_labels)

@app.post('/predict_detailed')
async def model_predict_detailed(req: PredictDetailedReq):
    granularity = 'coarse' if req.coarse_labels else 'fine'
    text = [x.text for x in req.data]
    """Predict with input"""
    output =  predict(model, text, cfg=cfg)
    results = aggregate_results(text, output, level=req.level, granularity=granularity, detailed_results=True)
    print(results)
    return {k:[x.tolist() for x in v] for k,v in results.items()}

cfg = read_config('./defaults.cfg')
model = instantiate_model(cfg)

if __name__ == "__main__":
  pass
