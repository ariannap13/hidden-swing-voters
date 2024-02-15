

import os
import sys
import json
import datetime

from functools import partial

from collections import Counter

import itertools

import tqdm

#import multiprocessing as mp
#import ray

from i3.nlp.textprocessor import TextProcessor
from i3.nlp.corpus.adapters import TxtAdapter
from i3.nlp.corpus import TextDoc

from i3.nlp.algorithms import WordCountAlgorithm

#ray.init(num_cpus=4)
#pool = mp.Pool(4)


class TextCorpus:

	""" Load, store and process a dataset """

	def __init__(self, filepath=None, file_adapter=TxtAdapter(), log_fd=sys.stderr, stream_func=None, preproc_fun=None):
		""" reads data if filepath is provided """
		self.filepath = filepath
		self.file_adapter = file_adapter
		self.log_fd = log_fd
		self.stream_func = stream_func
		self.preproc_fun = preproc_fun

		self.last_log_date = None
		self.start_log_date = None

		self.documents = []

		if self.filepath is not None:
			self.readCorpus()

	def preprocess(self, filepath=None, min_count=None):
		""" apply preprocessing steps to corpus, compute basic statistics, reads data if filepath provided """
		if filepath is not None:
			self.filepath = filepath
			self.readCorpus()

		self.normalizeCharacters()
		self.tokenize()
		self.stripPunctuation()
		self.filterStopWords()

		if min_count is not None:
			self.filterWordsByCount(min_count=min_count)

	# get documents

	def getDocuments(self):
		""" get list of TextDoc objects """
		return self.documents

	def getDocumentsTokens(self, join_tokens=False):
		""" get the tokens of the documents """
		return [doc.tokens if not join_tokens else TextProcessor.joinWords(doc.tokens) for doc in self.documents]

	def getDocumentsTexts(self):
		""" get the text of the documents """
		return [doc.text for doc in self.documents]

	# misc

	def size(self):
		""" get number of documents """
		return len(self.documents)

	# simple IO

	def saveTxt(self, fp, output_folder=".", process_tokens=True):
		""" save directly in a single file, one line per document """
		with open(os.path.join(output_folder, fp), "w") as f:
			for doc in self.documents:
				print(TextProcessor.joinWords(doc.tokens if process_tokens else doc.text), file=f)

	def loadTxt(self, fp, input_folder="."):
		""" load directly from a single file, one line per document """
		with open(os.path.join(input_folder, fp)) as f:
			self.documents = [TextDoc(x.rstrip("\n")) for x in f.readlines()]

	# advanced IO

	def addDoc(self, doc):
		doc = self.preproc_fun(doc) if self.preproc_fun else doc
		if self.stream_func is None:
			self.documents.append(doc)
		else:
			self.stream_func(doc)

	def addDocList(self, list_doc):
		for doc in list_doc:
			self.addDoc(doc)


	def readFile(self, filepath, doc_separator=None):
		""" process a single file with the file adapter """
		with open(filepath) as f:
			# one file = one document
			if doc_separator is None:
				doc = f.read().rstrip("\n")
				doc = TextDoc(doc)
				self.addDoc(doc)
			# one file = several documents
			else:
				list_docs = f.read().rstrip("\n").split(doc_separator)
				list_docs = [TextDoc(text) for text in list_docs]
				for doc in list_docs:
					self.addDoc(doc)

	def readCorpus(self, filepath=None, append=False):
		""" process corpus either enclose in a directory or in file with document separators """
		if filepath is not None:
			self.filepath = filepath

		self.logStep("reading corpus")

		if not append:
			self.documents = []

		if not os.path.exists(self.filepath):
			raise Exception("filepath does not exists: "+filepath)
		else:
			# process a single file
			if os.path.isfile(self.filepath):
				docs = self.file_adapter.readFile(self.filepath)
				self.addDocList(docs)
			# process all the files of the directory
			elif os.path.isdir(self.filepath):
				for fp in os.listdir(self.filepath):
					docs = self.file_adapter.readFile(os.path.join(self.filepath, fp))
					self.addDocList(docs)
			else:
				raise Exception("unsupported file type for: "+filepath)


	### Word filtering and transformation
	#
	# principles:
	#  - all function are applied on the corpus through either:
	#    - applyTransformationOnDocument
	#    - applyTransformationOnCorpus
	#
	# - the previous 2 functions take as an argument a function which is applied and which:
	#    - is static and defined in TextProcessor
	#    - if necessary is partially instanciated with specific arguments
	#    - is either applicable to list of string (transformWords*) or a string (other names)
	#
	# - the applied functions either:
	#    - modify corpus, and returns nothing (with replace=True)
	#    - left corpus untouched, and return the processed output
	#
	###

	# logging functions

	def logProgress(self, i=0, n=1000, end=False):
		""" log progress when processing a large corpus """

		if i % n == 0 or end:

			if end:
				i = self.size()

			# compute speed and remaining time
			#time_delta = (datetime.datetime.today() - self.last_log_date) if self.last_log_date is not None else None
			time_delta = (datetime.datetime.today() - self.start_log_date) if self.start_log_date is not None else None

			if self.start_log_date is None:
				self.start_log_date = datetime.datetime.today()

			self.last_log_date = datetime.datetime.today()

			if time_delta is None or time_delta.seconds == 0:
				speed_str = "n.a."
				remaining = "n.a."
			else:
				#speed = (n/(time_delta.seconds))
				speed = i / time_delta.seconds if time_delta.seconds > 0 else 0
				speed_str = ("%.2f" % speed) +" doc/s"
				remaining_time = (self.size() - i) / speed if speed > 0 else -1.
				if remaining_time <= 60:
					remaining = "%.2f" % (remaining_time)+" s"
				elif remaining_time <= 3600:
					remaining = "%.2f" % (remaining_time / 60)+" m"
				elif remaining_time <= 3600*24:
					remaining = "%.2f" % (remaining_time/3600)+" h"
				else:
					remaining = "%.2f" % (remaining_time/(3600*24))+" d"

			print("Processed:", i,
					"total:", self.size(),
					"%done:", ("%.2f" % (float(i)/self.size())),
					"-",
					"speed", speed_str,
					"remaining time", remaining,
					"-",
					datetime.datetime.today(),
					file=sys.stderr)
		return True

	def logStep(self, text):
		""" log information """
		if self.log_fd is not None:
			print("*", text, "-", datetime.datetime.today(), file=self.log_fd)

	# transform functions

	def applyTransformationOnDocument(self, doc, fun=lambda x: x, replace=True, process_tokens=True, process_doc=False, add_field=None):
		""" apply function on a single document
		
		fun: function to apply
		replace: in place replace
		process_tokens: if false process the original text, if true the tokens
		add_filed: add a new field and initialiaze it with the computed value
		log: whether to log progress on stdout
		"""
		return TextDoc.applyTransformationOnDocument(doc, fun=fun, replace=replace, process_tokens=process_tokens, process_doc=process_doc, add_field=add_field)

	def applyTransformationOnCorpus(self, fun, replace=True, process_tokens=True, process_doc=False, parrallel=False, log=True):
		""" apply a function on all the documents
		
		fun: function to apply
		replace: in place replace
		process_tokens: if false process the original text, if true the tokens
		log: whether to log progress on stdout
		"""

		def chunks(list, n):
			""" Yield successive chunks of size n """
			for i in range(0, len(list), n):
				yield list[i:i + n]

		# the way the logProgress function is called is a trick, to have faster code using list comprehension

		if True or not parrallel:
			res = [self.applyTransformationOnDocument(doc, fun, replace=replace, process_tokens=process_tokens, process_doc=process_doc) for i, doc in enumerate(self.documents) if not log or self.logProgress(i)]
		else:
#			self.logStep("parallel processing")
#			self.log_fd = None
			fun_apply = partial(TextDoc.applyTransformationOnDocument, fun=fun, replace=replace, process_tokens=process_tokens, process_doc=process_doc)
#			res = pool.map(fun_apply, self.getDocuments())


#			progress_bar = lambda x: x if parrallel else tqdm.tqdm

##			all_res = []
##			chunk_size=10
###			for chunk in tqdm.tqdm(chunks(self.documents, chunk_size), total=self.size()/chunk_size):
##			for chunk in chunks(self.documents, chunk_size):
##				all_res.append(pool.map(partial(self.applyTransformationOnCorpus, fun, replace, process_tokens, process_doc, False, log), chunk))
##			res = list(itertools.chain.from_iterable(all_res))

#			fun_apply = partial(self.applyTransformationOnDocument, fun=fun, replace=replace, process_tokens=process_tokens, process_doc=process_doc)
#			@ray.remote
#			def ray_func(x):
#				return self.applyTransformationOnDocument(x, fun=fun, replace=replace, process_tokens=process_tokens, process_doc=process_doc)
#
#			res_id = [ray.remote(fun_apply).remote(doc) for doc in self.getDocuments()]
#			res_id = [ray_func.remote(doc) for doc in self.getDocuments()]
#			res = ray.get(res_id)

#			self.log_fd = sys.stderr

#		self.logProgress(end=True)

		if replace:
			if process_doc:
				self.documents = res
			return None
		else:
			return res

	### Shortcuts for applying common corpus transformation

	def filterWords(self, word_list, fun=TextProcessor.filterStopWords, replace=True):
		""" mandatory argument: word_list: list of string """
		self.logStep("filter word list")

		fun_inst = partial(fun, word_list=word_list)
		return self.applyTransformationOnCorpus(fun_inst, replace=replace)

	def filterStopWords(self, fun=TextProcessor.filterStopWords, replace=True):
		""" filter stop words """
		self.logStep("filter stop words")

		return self.applyTransformationOnCorpus(fun, replace=replace)

	def filterWordsByCount(self, count=None, fun=TextProcessor.filterWordsByCount, min_count=0, max_count=None, replace=True):
		""" remove word with a low word count
		
		count: sklearn word count vector
		min_count: minimal count of a word to be included
		max_count: maximal coutn of a word to be included
		replace: if true, transform in place the document
		"""
		self.logStep("filter words by count")

		if count is None:
			count = WordCountAlgorithm().processCorpus(self)

		fun_inst = partial(fun, min_count=min_count, max_count=max_count, count=count)
		return self.applyTransformationOnCorpus(fun_inst, replace=replace)

	def normalizeCharacters(self, fun=TextProcessor.normalizeCharacters, replace=True):
		""" normalize unicode character representation """
		self.logStep("normalize characters")

		return self.applyTransformationOnCorpus(fun=fun, replace=replace, process_tokens=False)

	def stripPunctuation(self, fun=TextProcessor.transformWordsStripPunctuation, replace=True):
		""" remove punctuation """
		self.logStep("strip punctations")

		return self.applyTransformationOnCorpus(fun=fun, replace=replace)

	def normalizeSpacing(self, fun=TextProcessor.normalizeSpacing, replace=True, process_tokens=False):
		""" remove double spaces and empty lines """
		self.logStep("normaliz spacing")

		return self.applyTransformationOnCorpus(fun=fun, replace=replace, process_tokens=process_tokens)

	def tokenize(self, fun=TextProcessor.splitWords, tokenizer="split"):
		""" tokenize

		tokenizer: eithet "split" (faster) or "nltk"
		"""
		
		self.logStep("tokenize")

		fun_inst = partial(fun, tokenizer=tokenizer)
		res = self.applyTransformationOnCorpus(fun_inst, replace=False, process_tokens=False)
		for doc, tokens in zip(self.documents, res):
			doc.tokens = tokens

	##


if __name__ == '__main__':
	pass