

import os
import sys
import json
import datetime

from functools import partial

from collections import Counter

import itertools

import tqdm

import multiprocessing as mp
#import ray

from i3.nlp.textprocessor import TextProcessor
from i3.nlp.corpus.adapters import TxtAdapter
from i3.nlp.corpus import TextDoc, TextCorpus

from i3.nlp.algorithms import WordCountAlgorithm

#ray.init(num_cpus=4)
pool = mp.Pool(4)



def applyTransformationOnCorpus(corpus, fun, replace=True, process_tokens=True, process_doc=False, parrallel=True, log=True):
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

		if False and not parrallel:
			pass
#			res = [corpus.applyTransformationOnDocument(doc, fun, replace=replace, process_tokens=process_tokens, process_doc=process_doc) for i, doc in enumerate(self.documents) if not log or self.logProgress(i)]
		else:
#			self.logStep("parallel processing")
#			self.log_fd = None
			fun_apply = partial(corpus.applyTransformationOnDocument, fun=fun, replace=replace, process_tokens=process_tokens, process_doc=process_doc)
#			res = pool.map(fun_apply, corpus.getDocuments())




#			progress_bar = lambda x: x if parrallel else tqdm.tqdm

			all_res = []
			chunk_size=10
#			for chunk in tqdm.tqdm(chunks(self.documents, chunk_size), total=self.size()/chunk_size):
			for chunk in chunks(corpus.documents, chunk_size):
#				all_res.append(pool.map(fun_apply, chunk))
				all_res.append(pool.map(partial(TextCorpus.applyTransformationOnCorpus, fun, replace, process_tokens, process_doc, False, log), chunk))

			res = list(itertools.chain.from_iterable(all_res))

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
				corpus.documents = res
			return None
		else:
			return res

if __name__ == '__main__':
	pass