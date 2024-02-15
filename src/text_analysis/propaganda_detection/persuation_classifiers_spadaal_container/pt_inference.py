from pt_model import instantiate_model, predict
import pt_constants
from pt_postprocessing import aggregate_results, output_to_json
import pandas as pd
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
                        prog = sys.argv[0],
                        description = 'Inference script for multilabel PT model',
                        epilog = '_')
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', required=True)
    
    parser.add_argument('--model', required=False)
    parser.add_argument('--text_col',required=False, default = 'text')
    parser.add_argument('--guid_col',required=False, default = 'guid')
    parser.add_argument('--raw_output', required=False)
    parser.add_argument('--lang',required=False)
    args = parser.parse_args()

    model_fp = args.model
    text_fp = args.text
    text_col = args.text_col
    output_fp = args.output
    guid_col = args.guid_col
    raw_output = args.raw_output
    
    # load text
    df_txt = pd.read_csv(text_fp)
    if args.lang:
        df_txt = df_txt[df_txt.lang == args.lang]
        print('[*] Limiting to language: ', args.lang)
    
    list_texts = list(df_txt[text_col].values)
    list_guids = list(df_txt[guid_col].values) if guid_col in df_txt.columns else None
    
    # load model
    model = instantiate_model(model_fp) if model_fp else instantiate_model() #TODO replace
    
    if raw_output:
        raise NotImplementedError('Saving Raw output Not implemented yet.')
    
    else:
        output = predict(model, list_texts)
        output = aggregate_results(list_texts, output, level='sentence', augment_dict= True)
        #output = aggregate_results(text, output, level='sentence',coarse=True, augment_dict= True)
        output = aggregate_results(list_texts, output, level='paragraph', augment_dict= True)
        #output = aggregate_results(text, output, level='paragraph',coarse=True, augment_dict= True)
        
        dict_out = output_to_json(output, document_ids=list_guids)
        
        json_fp = open(output_fp, 'w')
        json.dump(dict_out, json_fp, indent = 4)
    # predict

    print("Inference completed")


if __name__ == "__main__":
    main()