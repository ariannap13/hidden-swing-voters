

import sys
import os
import fasttext

#PRETRAINED_MODEL_PATH = 'lid.176.bin'
PRETRAINED_MODEL_PATH = os.path.dirname(__file__)+"/../../../../python-emm-static-data/lid.176.bin"

TOPWORDS_PATH = os.path.dirname(__file__)+"/../../../../python-emm-static-data/corpus50x100_topwords/"

STATICDATA_PATH = os.path.dirname(__file__)+"/../../../../python-emm-static-data/" 


model = fasttext.load_model(PRETRAINED_MODEL_PATH)

def filter_lang(text, lang_to_keep):
	""" filter input based on required lang """

	if type(text) == str:
		return model.predict(text.strip())[0][0].split("__")[2] == lang_to_keep

	res = []
	for line in text:
		if model.predict(line.strip())[0][0].split("__")[2] == lang_to_keep:
			res.append(line)
	return res

def detect_lang(text, return_prob=False):
	""" detect lang of input """
	if type(text) == str:
		res = model.predict(text)[0][0].split("__")[2]
		#res = res[2] if not return_prob else [res[0], res[2]]
		return res
	return [ model.predict(line)[0][0].split("__")[2] for line in text]
#	return [ model.predict(line)[0][0].split("__")[2] if not return_prob else  model.predict(line)[0][0].split("__") for line in text]

def get_top_words(lang, script="default", include_count=False):

	fp = TOPWORDS_PATH + "topwords_" + lang + "_" + script + ".dat"

	if not os.path.exists(fp):
		raise Exception("topword file does not exist, try specifying script:" + fp)

	top_words = []

	with open(fp) as f:
		top_words = [ x if include_count else x.split()[0] for x in f.readlines()]

	return top_words

import pandas as pd

map_lang_stopwords = {}

def get_stopwords_corleone(lang):
	""" return stopwords based on Corleones's data """

	global map_lang_stopwords

	prefix = STATICDATA_PATH + "stopwords_corleone/"

	if not map_lang_stopwords:
		# reads csv with correspondance isocode <-> filename
		lang_items = list(pd.read_csv(prefix + "lang_corleone.csv").to_records(index=False))
		for filename, iso1, name, idx in lang_items:
			stop_words = []
			map_lang_stopwords[iso1] = [filename, stop_words]
			map_lang_stopwords[filename.lower()] = [filename, stop_words]
			map_lang_stopwords[filename.lower().capitalize()] = [filename, stop_words]
			
	if lang not in map_lang_stopwords:
		raise Exception("invalid language:" + str(lang))

	if not map_lang_stopwords[lang][1]:
		list_words = open(prefix + "files/" + map_lang_stopwords[lang][0]).read().split("\n")
		map_lang_stopwords[lang][1].extend(list_words)
	else:
		list_words = map_lang_stopwords[lang][1]

#	print(map_lang_stopwords)

	return list_words


def get_stopwords(lang, script=None, upper=True, capitalize=True):
	""" returns list of stop words for given language

	lang: language as 2 letter iso code ("fr") or english name ("french" or "French")
	script: optional, script in which the language is writtent
	upper: doubled the list with upper cased words
	capitalize: doubles the list with capitalized words
	"""

	list_words = get_stopwords_corleone(lang)

	list_words = [ x.strip() for x in list_words if "|" not in x and x.strip() != ""]

	stop_words = []

	stop_words += list_words

	if capitalize:
		stop_words += [x.capitalize() for x in list_words]

	if upper:
		stop_words += [x.upper() for x in list_words]

	return stop_words