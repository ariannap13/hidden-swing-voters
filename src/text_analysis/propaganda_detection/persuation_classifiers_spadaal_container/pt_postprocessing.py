import numpy as np
import pandas as pd
import re
from i3.nlp.textprocessor import TextProcessor
from more_itertools import flatten, windowed
from pt_constants import label_dict, label_dict_coarse, map_fineid2coarseid
import pt_constants
from typing import Union, List, Type
from enum import Enum
from nltk.tokenize import WordPunctTokenizer
from more_itertools import first, last, consecutive_groups


def postprocess_scores(output_raw: dict, threshold: Union[float, np.array], collapse_to_span=True):
    """
    Function processing raw XLM-R scores to two aligned labels, char_offsets (lists of np.arrays) given a threshold

    Arguments:
    - collapse_to_span: if set to True, offsets are maximally expanded to the (start,end) of the span's, rather than the token's

    Returns: dict {'labels': [np.arrays of labels], 'char_offsets': [lists of tuples (start,end) with char_offsets])

    """

    scores = output_raw['scores']
    offsets = output_raw['offsets']

    assert isinstance(threshold, (float, np.ndarray))

    if isinstance(threshold, (np.ndarray)):
        threshold_dim, pt_dimm  = threshold.shape[0], scores[0].shape[-1]
        assert  threshold_dim == pt_dimm

    offsets_list = list(map(np.array,offsets))
    offsets_lengths = list(map(len, offsets))

    # Group by document from array to list of arrays
    scores = np.split(scores,scores.shape[0]) 
    scores = [row[0,:len_] for row, len_ in zip(scores, offsets_lengths)] # Trim to len(offsets)

    positive_inds = [np.concatenate([(np.where(s > threshold))]).T for s in scores]
    labels = [s[:,1] for s in positive_inds] # Get only second columns (labels)

    if False:#collapse_to_span: #Off for now
        from more_itertools import unzip
        #FIX this apply function with map NOT LIKE THIS (use unzip or )
        char_offsets_aligned, labels = collapse_spans_row(positive_inds, offsets_list)

    else:
        # Align offsets with labels array
        char_offsets_aligned = [offs[inds[:,0]]  for inds, offs in zip(positive_inds, offsets_list)]

    return {'labels': labels, 'char_offsets': char_offsets_aligned}


def collapse_spans_row(res, segment_spans):
    """
    Automatically collapse spans when consecutive segments appear
    Input:
        - res: dict of the form {'label': [segments]}
        - segment_spans: list with (start, end) tuples of each segment
    """
    ret = {}
    for label, segments in res.items():
        if len(segments) > 1:
            collapsed = list( map(lambda x: (first(x),last(x, default=None)), (consecutive_groups(segments))))
            # Correct Nones
            collapsed = [(first,first) if last == None else (first, last) for (first, last) in collapsed ]

        else:
            collapsed = [(segments[0],segments[0])]

        ret[label] = [(segment_spans[first][0], segment_spans[last][1]) for (first, last) in collapsed ]

    return ret


def get_sent_offsets_row(text_row: str):
    """ 
    Given a string as input, segment into sentence and return their offsets
    Returns: [(start, end) ...]    
    """

    segments = TextProcessor.splitText(text_row, False, False, True, False, False, keep_indices=True)
    list_segments = TextProcessor.extract_segment_type(segments, "SENT:D", return_segment=True)

    sentence_offsets = [(segment.start, segment.end) for segment in list_segments]
    sentence_offsets[-1] = (sentence_offsets[-1][0], max(sentence_offsets[-1][0],len(text_row))) #expand last span to the end of document 

    return sentence_offsets


def get_par_offsets_row(text_row: str):
    """ 
    Given a string as input, segment into paragraph and return their offsets
    Returns: [(start, end) ...]
    """
    newline_offsets = [span.span() for span in re.finditer('\n\n[\n]*(?!$)',text_row)] # match any double (or more) newline except at the EOF
    paragraph_offsets= windowed([0] + list(flatten(newline_offsets)) + [len(text_row)], 2, step=2) # encode to spans of text

    return list(paragraph_offsets)


def get_word_offsets_row(text_row: str):
    """ 
    Given a string as input, segment into words and return their offsets
    Returns: [(start, end) ...]
    """
    span_generator = WordPunctTokenizer().span_tokenize(text_row)
    word_offsets= windowed(flatten(span_generator), 2, step=2) # encode to spans of text

    return list(word_offsets)


def agg_results_segments_row(token_char_offsets: np.array, labels: np.array, segments: list):
    """ Aggregate results on a given string given a list of segments 
    Input:
        - token_char_offsets : Nx2 (start, end) 
        - labels: array(N,)
        - segments: [(segment_start,segment_end), ... , ()]

    Returns: np.array: (segment_id, label_id)
    """
    segments_end = [s[1] for s in segments]
    #segments_right = [s[1] for s in segments]

    # Assign intervals to each token
    segment_inds_left = np.digitize(token_char_offsets[:,0], segments_end, right=False)
    segment_inds_right = np.digitize(token_char_offsets[:,1], segments_end, right=True)   

    ## COMMENTED OUT FOR NOW
    ## will be needed if char_offsets are not necessary subwords

    # Check if they are not consecutive, and include intermediate segments 
    # e.g. if we get spans [0,1,2] on left and [3,2,3] on right
    # we need to include [1,2] 
    left_join_right = np.vstack([segment_inds_left, segment_inds_right])
    #non_consecutive_mask = np.diff(left_join_right, axis=0) > 1


    segment_inds = np.hstack([segment_inds_left, segment_inds_right])
    labels = np.hstack([labels, labels])

    # Trim to max to avoid boundary errors
    segment_inds = np.clip(segment_inds,0, len(segments))

    # Deduplicate segments
    simplified_results = np.unique(np.vstack([segment_inds, labels]).T, axis=0) 
    return simplified_results


def process_labels_map(labels: list, mapping: dict):
    """ Transform labels using a mapping (e.g. fine to coarse)

    """
    func = np.vectorize(mapping.get)

    labels = map(lambda x:  func(x) if len(x) else x, labels)

    return list(labels)


def aggregate_results(text: list,
                      output: dict,
                      detailed_results=False,
                      augment_dict=False,
                      granularity='fine',
                      *,
                      level: str) -> dict:
    """
    Aggregate results on a given level of focus
    Arguments:
        - text: list of str
        - output: dict w/ elements:
            - char_offsets: list of Nx2 np.arrays of form: token x [start, end] offsets of each LABELED token
            - labels: list of Nx1 np.arrays with the LABEL of each token

    Optional Arguments:
        - detailed_results=False: results in the form
            ['labels'] : np.array (Nx1)
            {['segment_ids'] : np.array (Nx1) #aligned with labels
            ['segment_offsets'] : np.array (Mx1) #offset of every segment


        - augment_dict: copy input dict, and

        - granularity= ['fine', coarse]

    Returns:
        - Same format as the output if detailed_results is False

            dict {'labels': (np.array: shape(N,1) # with labels, 
                  'char_offsets': list : [(start,end), ... for each labeled],)}

        - dict {'labels': (np.array: shape(N,1) # with labels, 
                  'segment_offsets': list : [(start,end), ... for each segment_id],),
                  'segment_ids': list : np.array(Nx1) ... for each label],)}

    """
    assert 'char_offsets' in output.keys()
    assert 'labels' in output.keys()
    assert level in ['word','sentence', 'paragraph']
    assert granularity in ['coarse', 'fine']

    suffix = ''  # prefix of the dict keys

    char_offsets, labels = output['char_offsets'], output['labels']

    if granularity == 'coarse':
        labels = list(map(lambda x: np.vectorize(map_fineid2coarseid.get)(x) if len(x) else x, labels))  # replace fine with coarse idx
        suffix += '_coarse'

    if level == 'word':
        segments = map(get_word_offsets_row, text)
    if level == 'sentence':
        segments = map(get_sent_offsets_row, text)
    elif level == 'paragraph':
        segments = map(get_par_offsets_row, text)

    results = [(agg_results_segments_row(c, l, s), s) for c, l, s in zip(char_offsets, labels, segments)]

    segment_ids = [r[0][:, 0] for r in results]
    segment_offsets = [np.array(r[1]) for r in results]  # Not necessary to make them arrays, only for consistency
    labels = [r[0][:, 1] for r in results]

    ret = {}

    if detailed_results:
        ret['segment_ids'] = segment_ids
        ret['segment_offsets'] = segment_offsets
        ret['labels' + suffix] = labels

    else:
        try:
            char_offsets = [np.array([segment_offsets[row][id_] for id_ in ids]) for row, ids in enumerate(segment_ids)]
        except:
            print("segment_offsets: ", [(i, l) for i, l in enumerate(segment_offsets)])
            print("segment_ids: ", [(list(l), len(k)) for l, k in zip(segment_ids, segment_offsets)])

        ret['char_offsets'] = char_offsets
        ret['labels' + suffix] = labels

    if augment_dict:

        augmented = {}

        # Rename keys to exhibit level and add them
        for key in ret.keys():
            augmented[level + '_' + key] = ret[key]
        augmented.update(output)

        return augmented

    else:

        return ret


def output_to_json(output: dict, document_ids=None, map_to_labels=True, coarse_labels=False) -> dict:
    """
    Print JSON of results, if both fine and coarse labels exists it defaults to fine.
    Use coarse_labels=True to override this.
    Args:
        - output: dict with 'char_offsets' and ('labels' or 'labels_coarse') keys
        - document_ids: list of uids for each document to use as dict labels
        - map_to_labels: map labels to their string descriptions
        - coarse_labels: force to output coarse labels even if fine exist

    Returns: dict:{ (uid) -> [{'label':... , 'start': XX, 'stop': YY} ... ]}
    """

    assert 'char_offsets' in output.keys()
    assert 'labels' in output.keys() or 'labels_coarse' in output.keys()

    if coarse_labels or 'labels_coarse' in output.keys():

        assert 'labels_coarse' in output.keys()
        labels_key = 'labels_coarse'
        label_map = label_dict_coarse

    else:
        assert 'labels' in output.keys()
        labels_key = 'labels'
        label_map = label_dict

    df_ret = pd.DataFrame()

    for i, (offs, labels) in enumerate(zip(output['char_offsets'], output[labels_key])):
        if len(offs):
            doc_ids = np.full_like(labels,i)
            df_doc = pd.DataFrame(np.vstack([doc_ids.T, offs.T, labels.T ]).T, columns=['doc_id', 'start','end','label' ])
            df_ret = pd.concat([df_ret, df_doc])

    df_ret.label = df_ret.label.replace(label_map)

    if document_ids is not None:
        assert len(output[labels_key]) == len(document_ids)
        df_ret.doc_id = df_ret.doc_id.replace(dict(enumerate(document_ids)))
    else:
        document_ids = range(len(output['char_offsets']))  # in case of no uids just enumerate

    ret = df_ret.groupby('doc_id')[['label', 'start', 'end']].apply(lambda x: x.to_dict(orient='records')).to_dict()

    for i in filter(lambda x: x not in ret.keys(), document_ids):
        ret[i] = []

    return ret


class PersuationResults():
    "Wrapper class for results of Persuation Techniques multi-label, character level classifier results"

    def __init__(self, results, text: list, uids=None, token_offsets=None):
        """Initialize either from a dict directly from the classifier or from another PersuationResults object.
        Parameters:
        - text: list of strings
        - token_offsets: list of (start,end) tuples from the tokenizer (optional)
        """

        self.text = text
        self.labels_mapping = {'fine_id_to_coarse_id': pt_constants.map_fineid2coarseid}
        self.labels_enum = Enum('Persuation Techniques', list(pt_constants.label_dict.values()), start=0)

        if token_offsets:
            self.token_offsets = token_offsets

        if uids:
            self.uids = uids
        else:
            self.uids = range(len(text))

        if isinstance(results, PersuationResults):

                self.text = results.text
                self.uids = results.uids
                self.offsets = results.offsets
                self.labels = results.labels
                self.segment_ids = results.segment_ids
                self.labels_mapping = results.labels_mapping

        elif isinstance(results, dict):

            assert 'labels' in results.keys()
            assert 'char_offsets' in results.keys()

            self.offsets = {'char': results['char_offsets']}
            self.labels = {'char': results['labels']}
            self.segment_ids = {'char': [range(len(row)) for row in results['char_offsets']]} #edit case we have offsets given

            # Check if input has already encoded segments
            if 'sentence_offsets' in results.keys():
                self.offsets['sentence'] = results['sentence_offsets'] 
            if 'sentence_labels' in results.keys():
                self.labels['sentence'] = results['sentence_labels'] 
            if 'paragraph_offsets' in results.keys():
                self.offsets['paragraph'] = results['paragraph_offsets'] 
            if 'par_labels' in results.keys():
                self.labels['paragraph'] = results['par_labels'] 

    def __len__(self):
        return len(self.labels['char'])

    @property
    def output_full(self) -> dict:
        "Output the full results as stored internally"

        return {'labels': self.char_results['labels'],
                'char_offsets': self.char_results['char_offsets']}

    def output(self, level='char', granularity='fine', custom_map=None) -> dict:
        "Apply level and granularity aggregations"

        assert granularity in ['fine', 'coarse', 'custom']

        if level not in self.labels.keys():
            self.aggregate_results(levels=[level])

        labels = self.labels[level]

        if granularity == 'custom':
            assert custom_map in self.labels_mapping.keys()

        if granularity == 'coarse':
            mapping = self.labels_mapping['fine_id_to_coarse_id']
            #mapped_labels = {level: process_labels_map(labels, mapping) for level, labels in self.labels.items()}
            labels = process_labels_map(labels, mapping)

        # process to coarse
        return {'labels': labels,
               'char_offsets': self.offsets[level]}

    def __getitem__(self, idx: Union[List[int], slice, int]):
        """Overriding Indexing/Slicing functionality"""
        if type(idx) == slice:
            return self._get_slice(idx)

        elif type(idx) == list:
            return self._get_by_list(idx)

        elif type(idx) == int:
            return self._get_slice(idx)
        #TODO ADD Generator/Iterator arg type
        else:
            assert False  # replace

    def _get_slice(self, idx: Union[slice, int]):
        """Private method for slicing e.g. object[4:199]"""
        ret = PersuationResults(self, self.text)
        # Slice all arrays

        ret.offsets = {k:v[idx] for k,v in ret.offsets.items()}
        ret.labels = {k:v[idx] for k,v in ret.labels.items()}
        ret.segment_ids = {k:v[idx] for k,v in ret.segment_ids.items()}
        ret.text = ret.text[idx]

        return ret

    def _get_by_list(self, idx):
        """Private method for indexing using a list e.g. object[[1,2,6,466]]"""
        results = {k:[v[i] for i in idx] for k,v in self.char_results.items()}
        text = [text[i] for i in idx]
        ret = PersuationResults(results, text)

        if self.sentences:
            ref.sentences = [self.sentences[i] for i in idx]

        if self.paragraphs:
            ref.paragraphs = [self.paragraphs[i] for i in idx]

        return ret

    def aggregate_results(self, levels=['sentence','paragraph']):
        """Calculate aggregations on [sentence, paragraph] levels and store them inplace in the object attributes.
        Arguments:
        - levels: list of ['sentence', 'paragraph', 'word'] to calculate aggregations on.

        """

        char_results = {'labels': self.labels['char'],
                        'char_offsets': self.offsets['char']}

        #TODO FUTURE: detach granularity from level so that they are not computed twice
        for level in levels:
            agg_results = aggregate_results(self.text,
                                            char_results,
                                            detailed_results=True,
                                            augment_dict=False,
                                            level=level)

            self.offsets[level] = agg_results['segment_offsets']
            self.labels[level] = agg_results['labels']
            self.segment_ids[level] = agg_results['segment_ids']

        return self

    @property
    def sentences(self):
        """Return dict {uid: [offsets]} of all the char offsets for all sentences for all documents"""
        ret = map(np.ndarray.tolist, self.offsets['sentence'])
        ret = zip(self.uids, ret)
        #TODO guard that they exist
        #TODO make it return tuples
        return dict(ret)

    @property
    def words(self):
        """Return dict {uid: [offsets]} of all the char offsets for all sentences for all documents"""
        ret = map(np.ndarray.tolist, self.offsets['word'])
        ret = zip(self.uids, ret)
        #TODO guard that they exist
        #TODO make it return tuples
        return dict(ret)

    @property
    def paragraphs(self):
        """Return dict {uid: [offsets]}  of all the char offsets for all paragraph for all documents"""
        ret = map(np.ndarray.tolist, self.offsets['paragraph'])
        ret = zip(self.uids, ret)
        #TODO guard that they exist
        #TODO make it return tuples
        return dict(ret)

    @property
    def annotated_sentences(self):
        """Return dict {uid: [offsets]}  of all the char offsets for all sentences for all documents"""
        ret = map(np.ndarray.tolist, self.labels['sentence'])
        #TODO guard that they exist
        #TODO make it return tuples
        return list(ret)

    def to_dict(self,
                level='sentence',
                orient='segments',
                return_spans=False,
                collapse_spans=False):
        """Return JSON compatible dict of the form {uid: results } with the results 
        Arguments:
        - level: level of detail ['sentence', 'paragraph', 'word', 'char'].
        - orient: form of the returned dict
            if orient == 'labels' --> {'PT': ['setnence#1','sentence#2'...]}
            if orient == 'segments' --> {'sentence#1':['PT#1', 'PT#2']}
        - return_spans: return offsets in the form (start, end) instead of segment ids
        - collapse_spans: automatically collapse consecutive spans 
                        (e.g. if words [1,2,3,4] are annotated, return the span [start_of_word_1, end of word_4])
                        requires return_spans == True and orient == 'labels' to work
        """

        assert orient in ['segments', 'labels']
        assert level in ['sentence', 'paragraph', 'word', 'char']
        if collapse_spans:
            assert return_spans is True
            assert orient == 'labels'

        ret = {}

        iterator = zip(self.uids,
                       self.segment_ids[level],
                       self.labels[level],
                       self.offsets[level])

        for uid, segment_array, label_array, offsets in iterator:

            if return_spans and not collapse_spans:  # if collapse_spans == True the do this later
                segment_array = offsets[segment_array]
                tuple_gen = map(tuple, segment_array.tolist())
                segment_array = np.array([*tuple_gen], dtype='i,i')  # convert to array of tuples

            if orient == 'segments':
                main_orientation = segment_array
                nested_view = label_array
            else:
                main_orientation = label_array
                nested_view = segment_array

            # Groupby segment or label
            unique_vals, unique_idx = np.unique(main_orientation ,return_index=True)
            unique_idx = np.sort(unique_idx)  # sort them up

            grouped_by = np.split(nested_view, unique_idx[1:], axis=0)
            grouped_by_list = map(np.unique, grouped_by)  # deduplicate
            grouped_by_list = map(np.ndarray.tolist, grouped_by_list)

            if orient == 'labels':
                unique_vals = map(self.labels_enum, unique_vals)
            else:
                grouped_by_list = [list(map(self.labels_enum, entry)) for entry in grouped_by_list]
                unique_vals = map(tuple, unique_vals) if return_spans else unique_vals  # avoid 'writeable void-scalar' error

            result = {k: v for k, v in zip(unique_vals, grouped_by_list)}
            # Collapse spans in requested
            if collapse_spans:
                ret[uid] = collapse_spans_row(result, offsets)
            else:

                ret[uid] = result

        return ret