import numpy as np
import pandas as pd
import pickle
import ipywidgets as ipw
import json
from spacy import displacy 
import pt_constants
from ipywidgets import interact, Dropdown
from spacy import displacy
from matplotlib import pyplot as plt
from IPython.display import display, HTML

fine2shortnames = pt_constants.get_pt_nickname_converters('fine', use_st_format=True)[0]
shortnames2fine = pt_constants.map_nickname_stlabel
colors  = {'J:AA': '#fcba03',
'J:AF': '#fcba03',
'AR:AH': '#03bafc',
'J:APo': '#fcba03',
'C:AT': '#fcba03',
'J:AV': '#fcba03',
'S:CaO': '#b5fc03',
'S:CoO': '#b5fc03',
'C:CK': '#fcaaa4',
'AR:D': '#03bafc',
'MW:EM': '#d194f2',
'S:FDNC': '#b5fc03',
'J:FW': '#fcba03',
'AR:GA': '#03bafc',
'MW:LL': '#d194f2',
'AR:NCL': '#d194f2',
'MW:OVC': '#d194f2',
'AR:QR': '#03bafc',
'D:RH': '#fc7303',
'MW:R': '#d194f2',
'C:S': '#fcaaa4',
'D:SM': '#fc7303',
'D:W': '#fc7303'}
# TODO move to pt_constants.py
colors_full = {shortnames2fine[k]:v for k,v in colors.items()}

def PT_widget(df_text, results, uid_column='id'):
    
    df_text = df_text.set_index(uid_column)
    list_of_guids = list(results.keys())
    no_of_detections = list(map(len, results.values()))
    
    list_of_guids = [(str(guid) + '  ('+str(no)+')' , guid)
                     for no, guid  in sorted(zip(no_of_detections, list_of_guids), reverse=True)]
    
    document_select = ipw.Select(options=list_of_guids, description='guid (# detections)',
                                style= {'description_width': 'initial'},
                                layout={'width':'700px', 'height':'200px'})
    
    document_select.observe(lambda change: on_document_change(change))
    display(document_select)
    
    main_output = ipw.Output()
    display(main_output)

    def on_document_change(change):
        if change['name'] == 'value' and (change['new'] != change['old']):
            guid = change['new']

            results_doc = results[guid]
            text = df_text.loc[guid].text
            visualize_document(text, 
                               results_doc, 
                               guid, 
                               output=main_output)
    

    
def visualize_document(text, results, guid, output=None):
    "Display a full PT visualization tab given a document and the classifier results"
    #guid = str(df_text.id.iloc[idx])


    fine2shortnames = pt_constants.get_pt_nickname_converters('fine', use_st_format=True)[0]

    ents = [{'start':l['start'], 'end': l['end'], 'label': fine2shortnames[l['label']]} for l in  results]
    options = {"ents": colors.keys(), "colors": colors}
    doc = { 'ents': ents, 'text': text}
    
    # Setup Widgets
    sidepanel_layout = ipw.Layout(min_width='400px', align='right')
    panel_layout= ipw.Layout(width='1000px', align='left')
    out_left = ipw.Output()
    out_up = ipw.Output()
    out_down = ipw.Output()

    #Main widget
    pt_tab = ipw.Tab([ipw.HBox([ipw.Box([out_left],layout=panel_layout),
                                ipw.VBox([out_up, out_down],layout=sidepanel_layout)])])
    pt_tab.set_title(0, 'Persuation Techniques')
    
    #Dataframe with all neseccary metadata
    df_plot = pd.DataFrame(results)
    if not df_plot.empty:
        df_plot['span_text'] = df_plot.apply(lambda x: text[x.start:x.end], axis=1)
    else:
        df_plot = pd.DataFrame({'labels':[], 'span_text':[]})
        
    with out_up:

        display_detections_plot(df_plot, colors_full)

    with out_down:

        table = make_table_of_detections(df_plot)
        display(table)

    with out_left:
        displacy.render(doc, 
                        style='ent',
                        options=options,
                        manual=True ,
                        jupyter=True)
    
    if output:
        output.clear_output()
        with output:
            display(pt_tab)
    else:
        display(pt_tab)

def make_table_of_detections(df_plot, wrap_len=50) -> ipw.HTML:
    "Display list of PT detections. Given a DataFrame with ['label','span_text'] columns, return an HTML with all the detected spans"
    if not df_plot.empty:
        assert 'span_text' in df_plot.columns
        assert 'label' in df_plot.columns


        # Wrap long Spans
        df_plot['Span Text'] = df_plot['span_text'].str.wrap(wrap_len)
        # Construct table form
        to_display = df_plot[['label','Span Text']].sort_values('label').set_index('label')
        to_display = HTML(to_display.to_html().replace("\\n","<br>"))
    
    else:
        to_display = HTML(' ')
    
    return to_display

def display_detections_plot(df_plot, colors_full):
    "Display a detections plot w/ frequency of each detected PT in the document"
    
    if not df_plot.empty:
        assert 'label' in df_plot.columns

        detections_per_label = df_plot['label'].value_counts()
        bar_colors = [colors_full[ind] for ind in detections_per_label.index]
        ax = detections_per_label.plot(kind='bar', color=bar_colors)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
    
    else:

        plt.text(0.5, 0.5, "No persuasion found", size=25, rotation=0.,
                 ha="center", va="center",
                 bbox=dict(boxstyle="round",
                           ec=(0.3, 0.3, 0.3),
                           fc=(0.9, 0.9, 0.9),
                           )
                 )
        
    plt.ylabel('# of Detections')
    plt.show()