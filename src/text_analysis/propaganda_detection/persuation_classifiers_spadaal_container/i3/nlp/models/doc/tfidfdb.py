
import json

from typing import List, Dict, Tuple

from i3.nlp.algorithms import TfidfAlgorithm

from i3.nlp.models.doc import VectorDB

class TFIDFVectorDB(VectorDB):

	def __init__(self, min_tfidf=0.0, tfidf_vectorizer=None):
		self.min_tfidf = min_tfidf
		self.tfidf_algo = TfidfAlgorithm(tfidf_vectorizer)

		self.map_word_tfidf = None

		if tfidf_vectorizer is not None:
			self.map_idx_words = tfidf_vectorizer.get_feature_names()		
#			self.buildMap()

	def indexCorpus(self, corpus, process_tokens=True):
		
		self.tfidf_algo.processCorpus(corpus, process_tokens=process_tokens)

		self.map_idx_words = self.tfidf_algo.tfidf_vectorizer.get_feature_names()

		self.buildMap()

	def buildMap(self):
		""" build map word to tfidf"""

		tfidf_values = self.tfidf_algo.tfidf_matrix.tocoo()

		map_wordidx_tfidf: Dict[List, float] = {k:v for k,v in zip(tfidf_values.col, tfidf_values.data) if v >= self.min_tfidf}

		self.map_idx_words = self.tfidf_algo.tfidf_vectorizer.get_feature_names()

		res: Dict[str, float] = {}
		for word_vocab_idx, tfidf in map_wordidx_tfidf.items():
			if tfidf >= self.min_tfidf:
				word = self.map_idx_words[word_vocab_idx]
				res[word] = tfidf
		self.map_word_tfidf = res

	def filterWordsByTfidf(self, list_words: List[str], min_tfidf: float=0.1) -> List[Tuple[str,float]]:
		""" filter word list by tfidf """
		#tfidf_values = self.tfidf_algo.tfidf_vectorizer.transform(list_words)
		#tfidf_values = tfidf_values.tocoo()
		#map_word_tfidf: Dict[List, float] = {k:v for k,v in zip(tfidf_values.col, tfidf_values.data)}

		map_word_tfidf = { w:self.map_word_tfidf[w] for w in list_words if w in self.map_word_tfidf}

		res: List[Tuple[str, float]] = []
#		for word_vocab_idx, tfidf in map_word_tfidf.items():
		for word, tfidf in map_word_tfidf.items():
			if tfidf >= min_tfidf:
				#word = self.map_idx_words[word_vocab_idx]
				res.append((word, tfidf))
		return res

	def getDocumentVector(self, doc=None, doc_id: int=None, min_tfidf: float=None, process_tokens=True) -> List[Tuple[str, float]]:

		min_tfidf = self.min_tfidf if min_tfidf is None else min_tfidf

		doc_tfidf: List[Tuple[str, float]]

		words = doc.tokens
		#doc_tfidf = self.tfidf_algo.tfidf_vectorizer.transform(words)
		doc_tfidf = self.filterWordsByTfidf(words, 0.)
	
#		doc_tfidf = [ (w, self.map_word_tfidf[w]) for w in set(doc.tokens) if w in self.map_word_tfidf ]
		doc_tfidf = [ (w,v) for w,v in doc_tfidf if v >= min_tfidf]
		doc_tfidf.sort(key=lambda x: x[1], reverse=True)
		#doc_tfidf = doc_tfidf[:100]

		res = doc_tfidf


##		print("tfidf:process doc: ", "doc " + str(doc) if doc is not None else "doc_id "+str(doc_id))
#
#		tfidf_values = None
#
#		if doc is not None:
#			# compute tfidf representation for the document
##			list_words = doc if type(doc) == list else doc if type(doc) == str else None
#			list_words = doc.tokens if process_tokens else doc.text.split()
##			print("LW", list_words)
#			tfidf_values = self.tfidf_algo.tfidf_vectorizer.transform(list_words)
#			tfidf_values = tfidf_values.tocoo()
#		elif doc_id is not None:
#			# fetch previously computed representation for the document
#			tfidf_values = self.tfidf_algo.tfidf_matrix[doc_id,:].tocoo()
#		else:
#			raise Exception("document is None")
#
#		# recover words
#		map_word_tfidf: Dict[List, float] = {k:v for k,v in zip(tfidf_values.col, tfidf_values.data)}
#		
##		print("tfidf:iterate", len(map_word_tfidf))
#
#		map_idx_words = self.tfidf_algo.tfidf_vectorizer.get_feature_names()
#
#		res: List[str] = []
#		for word_vocab_idx, tfidf in map_word_tfidf.items():
#			if tfidf >= min_tfidf:
#				word = map_idx_words[word_vocab_idx]
#				res.append((word, tfidf))
#		res.sort(key=lambda x: x[1], reverse=True)
#
##		print(res)
#
		return res


	def searchSimilar(self, doc):
		raise Exception("non implemented method")

	def saveDB(self, fp):
		with open(fp, "w") as f:
			json.dump(self.map_word_tfidf, f)

	def loadDB(self, fp):
		with open(fp) as f:
			self.map_word_tfidf = json.load(f)


if __name__ == '__main__':
	pass

#import os
#
#from scipy import spatial, sparse
#
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import CountVectorizer
#
#from i3.nlp.models.doc import VectorDB
#
##########
## TFIDF #
##########
#
#class TFIDFVectorDB(VectorDB):
#
#	def __init__(self):
#
#		self.count_vectorizer = None
#		self.tfidf_transformer = None
#	
#		self.word_count_vector = None
#
#	# interface methods
#
#	def indexCorpus(self, corpus):
#		""" compute TFIDF for all documents of corpus """
#
#		self.computeTFIDF(corpus)
#
#	def getDocumentVector(self, doc):
#		""" return TFIDF vector of given document"""
#
##		print("getTI:", doc)
#
#		if type(doc) == int:
##			print("A")
#			doc_idx = doc
#			repr_tfidf = self.tfidf_matrix[doc_idx,:]
##			print(repr_tfidf.shape)
#		else:
#			#print("B")
#			#print(doc)
#			#print(self.count_vectorizer.vocabulary_)
#			#print(self.count_vectorizer.transform([doc]))
#			#query_count = self.count_vectorizer.transform([doc])
##			#repr_tfidf = self.tfidf_transformer.transform(self.count_vectorizer.transform(self.word_count_vector))
#			repr_tfidf = self.tfidf_transformer.transform(self.count_vectorizer.transform([doc]))
#
##		print("reprTI:", repr_tfidf)
#
#		repr_coo = repr_tfidf.tocoo()
#
#		items = []
#
##		list_absent = []
#
#		map_words = self.tfidf_transformer.get_feature_names()
#
#		print(list(zip(repr_coo.row, repr_coo.col, repr_coo.data)))
#
#		for _,j,v in sorted(zip(repr_coo.row, repr_coo.col, repr_coo.data), key=lambda x: x[2], reverse=True):
#			print(j)
#			word = map_words[j]
#			items.append((word, v))
##			if word in self.word_embedding.wv.vocab:
##				items.append((word, v))
##			else:
##				list_absent.append(word)		
##		print(list_absent)
#
##		print("itemsTI:", items)
#
#		return sorted(items,key=lambda x: x[1], reverse=True)
#
#	# ad hoc methods
#
#	def computeWordCount(self, corpus, min_count=0, max_count=None):
#		""" compute TF """
#
#		self.count_vectorizer = CountVectorizer(lowercase=False, min_df=10)
#		self.word_count_vector = self.count_vectorizer.fit_transform(corpus.getDocumentsTexts())
#
#		print("tot words:", self.word_count_vector.shape)
#
#		sum_words = self.word_count_vector.sum(axis=0).tolist()[0]
#		word_count =  zip(self.count_vectorizer.get_feature_names(), sum_words) #[(word, sum_words[0, idx]) for word, idx in self.count_vectorizer.vocabulary_.items()]
#		word_count = [ (w,v) for w,v in word_count if v >= min_count and (v <= max_count if max_count is not None else True)]
#		word_count = sorted(word_count, key = lambda x: x[1], reverse=True)
#
#		self.word_count = word_count
#
#		return word_count
#
##		return sorted([ (self.count_vectorizer.get_feature_names()[w_idx], v) for w_idx,v in
##				zip(repr_coo.row, repr_coo.data)
##				if v >= min_count and (v <= max_count if max_count is not None else True)],
##			key=lambda x: x[1], reverse=True)
##		for i,j,v in zip(repr_coo.row, repr_coo.col, repr_coo.data):
##			items.append((self.count_vectorizer.get_feature_names()[j], v))
#
#	def computeTFIDF(self, corpus):
#		""" computes TFIDF """
#		
##		print(corpus.getDocumentsTexts())
#
#		self.computeWordCount(corpus)
#
#		self.tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
#		self.tfidf_transformer.fit(self.word_count_vector)
##		self.tfidf_transformer.fit(np.array(corpus.getDocumentsTexts(), dtype=str).reshape(-1,1))
##		self.tfidf_transformer.fit([[x] for x in corpus.getDocumentsTexts()])
#
#		db_filename = corpus.filepath+"__tfidfz.npz"
#		if os.path.exists(db_filename):
#			self.tfidf_matrix = sparse.load_npz(db_filename)
#			return self.tfidf_matrix
#
#		self.tfidf_matrix = self.tfidf_transformer.transform(self.word_count_vector)
##		self.tfidf_matrix = self.tfidf_transformer.transform(corpus.getDocumentsTexts())
##		self.tfidf_matrix = self.tfidf_transformer.transform([[x] for x in corpus.getDocumentsTexts()])
#
##		print("AAAA", self.tfidf_matrix)
#
#		sparse.save_npz(db_filename, self.tfidf_matrix)
#		
#		return self.tfidf_matrix
#