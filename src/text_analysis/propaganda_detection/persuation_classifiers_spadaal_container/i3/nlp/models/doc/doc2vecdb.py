
import os
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument


from i3.nlp.models.doc.vectordb import VectorDB
from i3.nlp.models.doc.tfidfdb import TFIDFVectorDB


###########
# Doc2Vec #
###########


class Doc2VecEmbeddingDB(VectorDB):

	def __init__(self, tfidf_db, doc2vec_fp=None):#: TFIDFVectorDB):

		self.tfidf_db = tfidf_db #: TFIDFVectorDB = tfidf_db

		self.model_doc2vec = None

		if doc2vec_fp is not None:
			self.model_doc2vec = Doc2Vec.load(doc2vec_fp)
			print("MODEL", self.model_doc2vec)

	def indexCorpus(self, corpus):

		db_filename = corpus.filepath+"__emb.doc2vec"

		if os.path.exists(db_filename):
			print("log : loading db")
			#with open(db_filename, "rb") as f:
			#	self.db_embeddings = pickle.load(f)
			self.model_doc2vec = Doc2Vec.load(db_filename) 
			return

		documents = [TaggedDocument(doc.split(" "), [i]) for i, doc in enumerate(corpus.getDocumentsTexts())]

		self.model_doc2vec = Doc2Vec(documents, steps=20, vector_size=20, window=5, min_count=1, workers=7)

		print("log : saving db")

		self.model_doc2vec.save(db_filename)


	def getDocumentVector(self, doc):

		if type(doc) == int:
			doc_idx = doc
			return self.model_doc2vec.infer_vector(doc_idx)

		if isinstance(doc, np.array.__class__):
			return doc

#		doc_words = doc.split()

		doc_words = [ w for w,_ in self.tfidf_db.getDocumentVector(doc)]

#		doc_words = [ w for w, _ in self.tfidf_db.getDocumentVector(0)|]


		print("QUERY: ", doc_words)


#		print(sorted(self.tfidf_db.getDocumentVector(0)))

		vector = self.model_doc2vec.infer_vector(doc_words)

		print(vector)

		return vector


	def searchSimilar(self, doc, max_res=10):
		


#		if type(doc) == int:
#			doc_idx = doc
#			doc = self.corpus.documents[doc_idx]
#		
#		doc_words = []
#		for w in doc.split(" "):
#			if w in self.model_doc2vec.wv.vocab:
#				doc_words.append(w)
#			else:
#				print("OOV", w)


#		for w in self.model_doc2vec.wv.vocab:
#			print(w)


#		print(self.model_doc2vec.docvecs.most_similar(positive=doc_words) )
#		lll

		vector = self.getDocumentVector(doc)
		
		res = self.model_doc2vec.docvecs.most_similar([vector], topn=max_res) 

		return res

#		return self.model_doc2vec.docvecs.most_similar(0) 


#		return self.model_doc2vec.most_similar(positive=doc_words)

