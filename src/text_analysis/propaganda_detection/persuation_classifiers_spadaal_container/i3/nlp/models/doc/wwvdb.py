

import os
import numpy as np
import json
from scipy import spatial

from functools import partial

from i3.nlp.models.doc.vectordb import VectorDB

from i3.nlp.corpus import TextDoc, TextCorpus

from gensim.models import KeyedVectors, Word2Vec

from i3.nlp.models.doc.tfidfdb import TFIDFVectorDB	


from collections import Counter

########################
# Weighted Word Vector #
########################

class WeightedWordVectorEmbeddingDB(VectorDB):

	def __init__(self, word_embedding, map_word_tfidf):

		self.word_embedding = word_embedding
		#self.tfidf_db = tfidf_db
		self.map_word_tfidf = map_word_tfidf

	# interface methods

	def indexCorpus(self, corpus):

		self.db_embeddings = []
#		self.map_idx_emb = {}
		#self.matrix_emb = np.zeros(shape=(corpus.size(), ))

#		for i in range(corpus.size()):
#			if i % 1000 == 1:
#				print("processed", i)
#
#			emb = self.getDocumentVector(i, use_caching=False)
#			self.db_embeddings.append((i, emb))
#			self.map_idx_emb[i] = emb

#		import i3.nlp.corpus.parallelexec 


#		corpus = TextCorpus()
#		corpus.documents = [TextDoc("aaa", tokenize=True)]*20

#		list_emb = i3.nlp.corpus.parallelexec.applyTransformationOnCorpus(corpus, fun=self.getDocumentVector, replace=False, process_doc=True)

		partial_fun = partial(self.getDocumentVector, min_tfidf=0.01, max_words=200)

		list_emb = corpus.applyTransformationOnCorpus(fun=partial_fun, replace=False, process_doc=True)

	#	self.map_idx_emb = {i:emb for i,emb in enumerate(list_emb)}
		self.db_embeddings = [ (i,emb) for i,emb in enumerate(list_emb)]

#			print(emb)

#			np.savetxt(db_filename, self.map_idx_emb.values())

		#print(self.db_embeddings[:10])


	def getDocumentVectorById(self, doc_id=None, use_caching=True):

		doc_idx = doc_id
		
		if use_caching:
			doc_emb = self.db_embeddings[doc_idx][1]
			return doc_emb

#	def getDocumentVector(self, doc=None, weights=None, min_tfidf: float=0.0, max_words=None, use_caching=True):
#		self.tfidf_db = None
#		self.word_embedding = None
#		return None

	def getDocumentVector(self, doc=None, weights=None, min_tfidf: float=0.0, max_words=10, use_caching=True):

#		self.word_embedding = Word2Vec.load(
#			"/media/me/281ceffc-389d-410a-ac63-735e64ea3f2b/Data/Home/Work/Projects/" +
#			"SETA/SeTA_CORD/ModelsLucia/cord-19-models/wv-sg0hs1.bin"
#			) 

#		corpus = TextCorpus("/media/me/281ceffc-389d-410a-ac63-735e64ea3f2b/Data/Home/Work/Projects/PythonEMM/Tests/devtest_data/nanocorpus.txt")

#		self.tfidf_db = TFIDFVectorDB(0.05)
#		self.tfidf_db.indexCorpus(corpus, process_tokens=True)


#		print(doc)
#		print(self.matrix_emb.shape)

#		if isinstance(doc, np.array.__class__):
#			return doc

		# extract vectors of words

		#print(doc)

		#print(list(self.map_word_tfidf.items())[:100])

		list_pairs = []

		count_weights = 0

		if weights is None:
#			print("wwv:get tfidf")

			doc_tfidf = None
#			if type(doc) != int:
#				doc_tfidf = self.tfidf_db.getDocumentVector(doc)
#			else:
#				doc_tfidf = self.tfidf_db.getDocumentVector(doc_id=doc)

	#		print("AAA", doc)

			doc_tfidf = [ (w, self.map_word_tfidf[w]) for w in set(doc.tokens) if w in self.map_word_tfidf]
			doc_tfidf.sort(key=lambda x: x[1], reverse=True)
			doc_tfidf = doc_tfidf[:max_words]

			counter = Counter(doc.tokens)

#			print("collect embeddings")

			for word, tfidf in doc_tfidf:
				if word not in self.word_embedding:
					continue
				if tfidf < min_tfidf:
					continue
				weight = tfidf * counter[word]
				count_weights += counter[word]
	#			print("W", word, weight, tfidf, counter[word])
				word_vect = self.word_embedding[word]
				if word_vect is not None:
					list_pairs.append((word_vect, weight))
		else:
			for word, weight in weights.items():
				if word not in self.word_embedding:
					print("NOT IN VOC", word)
					continue
				word_vect = self.word_embedding[word]
				if word_vect is not None:
					list_pairs.append((word_vect, weight))
					count_weights += 1

#		print("filter embeddings")

		# filter highest tfidf
	#	list_pairs.sort(key=lambda x: x[1])
		list_pairs = list_pairs[:max_words if max_words is not None else len(list_pairs)]

#		print("sum embeddings")

		# sum vectors
		norm_count = 0.
		vect = []
		for word, weight in list_pairs:
			vect.append(word_vect * weight)
			norm_count += weight 

		# normalize
		if len(vect) > 0:
			vect = np.sum(vect, axis=0) / norm_count # = np.mean(vect, axis=0) / count_weights # norm_count
			norm2 = np.linalg.norm(vect, axis=0)
			vect /= norm2
		else:
			vect = None

	#	if vect is not None:
	#		print(vect[:10])
	#	else:
	#		print("None")

	#	if vect is not None:
	#		print( np.linalg.norm(vect, axis=0), np.linalg.norm(vect))


		return vect

	def searchSimilar(self, doc, weights=None, max_docs=10):

		if weights is None:
			weights = {w:1. for w in doc.tokens}

		query_emb = self.getDocumentVector(doc, weights=weights)

		print("Q ", np.linalg.norm(query_emb, axis=0))

#		print(query_emb)

#		results = [ (doc_id, 1. - spatial.distance.cosine(query_emb, doc_emb)) for doc_id, doc_emb in self.db_embeddings if doc_emb is not None]
#		results = sorted(results, key=lambda x: x[1], reverse=True)

		
		results = [ (doc_id, 1. - spatial.distance.cosine(query_emb, doc_emb)) for doc_id, doc_emb in self.db_embeddings if doc_emb is not None]
		results = sorted(results, key=lambda x: x[1], reverse=True)

		print(results[:10])
		print(results[-10:])

#		doc = self.corpus.documents[doc_idx]
#		print("TOP")
#		for doc_idx, doc_sim, in results[:10]:
#			print("%.3f" % doc_sim, "\t",self.corpus.documents[doc_idx][:200])
#
#		print()
#		print("LAST")
#		for doc_idx, doc_sim, in results[-10:]:
#			print("%.3f" % doc_sim, "\t", self.corpus.documents[doc_idx][:200])

		return results[:max_docs]


	def saveDB(self, fp):
		with open(fp, "w") as f:
			try:
				data = [ (idx, [float(x) for x in vect] if vect is not None else "NONE") for idx,vect in self.db_embeddings]
			except:
				print("ERROR", self.db_embeddings)
			json.dump(data, f)

	def loadDB(self, fp):
		with open(fp) as f:
			data = json.loads(f.read())
			self.db_embeddings = [ (idx, np.array(vect) if vect != "NONE" else None) for idx, vect in data]
	
if __name__ == '__main__':
	pass