
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

from .algorithm import CorpusAlgorithm


class TfidfAlgorithm(CorpusAlgorithm):

	def __init__(self, tfidf_vectorizer=None):
		super(TfidfAlgorithm, self).__init__()

		self.tfidf_vectorizer = tfidf_vectorizer

#		if tfidf_vectorizer is not None:
#			self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)

	def processCorpus(self, corpus, process_tokens=True):
		""" computes TFIDF """

		self.logStep("compute IDF")
		
#		if self.word_count is None:
#			self.computeWordCount()

		documents = corpus.getDocumentsTexts() if not process_tokens else corpus.getDocumentsTokens(join_tokens=True)
		
#		db_filename = corpus.filepath+"__tfidfz.npz"
#		if os.path.exists(db_filename):
#			self.tfidf_matrix = sparse.load_npz(db_filename)
#			return self.tfidf_matrix

		self.tfidf_vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, sublinear_tf=True)
		self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
#		sparse.save_npz(db_filename, self.tfidf_matrix)
		
		return self.tfidf_matrix