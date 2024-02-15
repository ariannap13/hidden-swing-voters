

from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

from .algorithm import CorpusAlgorithm

class WordCountAlgorithm(CorpusAlgorithm):

	def __init__(self):
		super(WordCountAlgorithm, self).__init__()
		#
		self.count_vectorizer = None
		self.word_count_vector = None
		self.word_count = None

	def processCorpus(self, corpus, min_count=0, max_count=None, process_tokens=True):
		""" compute TF """

		self.logStep("compute TF")
		
		self.count_vectorizer = CountVectorizer(lowercase=False) #, min_df=20)

		documents = corpus.getDocumentsTexts() if not process_tokens else corpus.getDocumentsTokens(join_tokens=True)

		self.word_count_vector = self.count_vectorizer.fit_transform(documents)

		sum_words = self.word_count_vector.sum(axis=0).tolist()[0]
		word_count =  zip(self.count_vectorizer.get_feature_names(), sum_words) #[(word, sum_words[0, idx]) for word, idx in self.count_vectorizer.vocabulary_.items()]
		word_count = { w: v for w,v in word_count if v >= min_count and (v <= max_count if max_count is not None else True) }

		self.word_count = Counter(word_count)

		return word_count
