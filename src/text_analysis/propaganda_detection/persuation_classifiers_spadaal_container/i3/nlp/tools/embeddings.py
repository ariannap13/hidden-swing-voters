
from i3.nlp.tools.language import detect_lang

class Vectorizer:

	def __init__(self, model_type, model_fp=None):
		self.model_type = model_type
		self.model_fp = model_fp
		self.model = None

		if model_type == "laser":
			from laserembeddings import Laser
			self.model = Laser()
		elif model_type == "bert":
			from sentence_transformers import SentenceTransformer
			model_fp = 'bert-base-nli-stsb-mean-tokens' if model_fp is None else model_fp
#			model_fp = 'xlm-r-100langs-bert-base-nli-stsb-mean-tokens' if model_fp is None else model_fp
			self.model = SentenceTransformer(model_fp)
		else:
			raise Exception("invalid method")
            
	def vectorizeOne(self, text, lang=None):
		vector = None
		if self.model_type == "laser":
			if lang is None:
				lang = detect_lang(text)
				#raise Exception("Missing lang argument for LASER")
			vector = self.model.embed_sentences([text], lang)
		elif self.model_type == "bert":
			vector = self.model.encode([text])
		else:
			raise Exception("invalid method")
#		vector = normalize(vector)
		return vector[0]

	def vectorizeAll(self, list_text, lang=None):
		list_vectors = None
		if self.model_type == "laser":
			if lang is None:
				#print("Warning, Missing lang argument for LASER")
				raise Exception("Missing lang argument for LASER")
			list_vectors = self.model.embed_sentences(list_text, lang)
		elif self.model_type == "bert":
			list_vectors = self.model.encode(list_text)
		else:
			raise Exception("invalid method")
#		list_vectors = [normalize(v.reshape(1,-1))[0] for v in list_vectors]
		return list_vectors

	def vectorizeAllChunks(self, list_text, chunk_size=4096, lang=None):
		def chunks(l, n):
			for i in range(0, len(l), n):
				yield l[i:i + n]

		list_vectors = []

		for chunk_texts in chunks(list_text, chunk_size):
			chunk_vectors = self.vectorizeAll(chunk_texts, lang)

			list_vectors.append(chunk_vectors)
		
		return list_vectors

from scipy.spatial.distance import cosine

def distance(vect1, vect2):

	dist_sem = cosine(vect1, vect2)

	return dist_sem

def similarity(vect1, vect2):

	dist_sem = 1. - cosine(vect1, vect2)

	return dist_sem