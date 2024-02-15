
from i3.nlp.corpus import TextCorpus

class VectorDB:

	def __init__(self):
		pass

	def indexCorpus(self, corpus, process_tokens=True):
		raise Exception("abstract method")

	def getDocumentVector(self, doc):
		raise Exception("abstract method")

	def searchSimilar(self, doc, max_res):
		raise Exception("abstract method")

	def saveDB(self, fp):
		raise Exception("abstract method")

	def loadDB(self, fp):
		raise Exception("abstract method")
