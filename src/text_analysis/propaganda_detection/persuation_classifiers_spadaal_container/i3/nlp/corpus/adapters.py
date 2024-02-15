
import json

from i3.nlp.corpus.textdoc import TextDoc

# adapter to parse different file formats directly to a TextDoc

class FileAdapter:

	def __init__(self):
		pass

	def parseFile(self, fp):
		raise Exception("abstract method")

#		with open(fp) as f:
#			return self.parseContent(fp.read())

#	def parseContent(self, content):
#		raise Exception("abstract method")


class JsonAdapter(FileAdapter):

	def __init__(self, path_title=None, path_text=None, path_tag=None, doc_separator="\n", extract_title=False, extract_text=True, stacked_json=False, starting_level=[], return_json=False):
		"""
		Transform a json file into a set of TextDoc

		path_tag: dictionary with a set of pairs (tag, path to extract information to go under this tag)
		extract_*: which data to extract and setup as main text
		stacked_json: if a file a on a each record (line) a different json
		
		TODO refactor to remove ad-hoc paths (title, and text), everythin shoould ba a dict or a structured documents
		"""

		self.path_text = path_text
		self.path_tag = path_tag
		self.doc_separator = doc_separator
		self.path_title = path_title
		# TODO: remove these and make them implicit
		self.extract_tag = path_tag is not None
		self.extract_title = path_title is not None #extract_title
		self.extract_text = path_text is not None #extract_text
		self.stacked_json = stacked_json
		self.starting_level = starting_level
		self.return_json = return_json

	def getPath(self, path, data):
		""" extract a path from a parsed json content """
		for p in path:
			if p in data:
				data = data[p]
			else:
				return None
		return data

	def parseDoc(self, data):
		""" extract the required information from a json object """

		if self.return_json:
			return data

		text = []
		tags = {}

		# extract text
		if self.extract_title:
			res = self.getPath(self.path_title, data)
			if res is not None:
				text.append(res)
		if self.extract_text:
			res = self.getPath(self.path_text, data)
			if res is not None:
				text.append(res)
		text = "   ".join(text)
		# extract tags
		if self.path_tag:
			for key, path in self.path_tag.items():
				res = self.getPath(path, data)
				tags[key] = res

		return TextDoc(text, tags)

	def readFile(self, fp):
		# uses a generator, to make it possible to stream the processing,
		# therefore being able to processing very large corpus
		with open(fp) as f:
			# one file = one document
			if self.doc_separator is None:
				d = json.load(f)
				if self.starting_level:
					d = self.getPath(self.starting_level, d)
				doc = self.parseDoc(d)
				yield doc
			# one file = several documents
			else:
				if self.stacked_json:
					for line in f:
						doc = self.parseDoc(json.loads(line))
						yield doc
				else:
					data = json.load(f)
					if self.starting_level:
						data = self.getPath(self.starting_level, data)
					for d in data:
						doc = self.parseDoc(d)
						yield doc
	
	def readText(self, text):
		# TODO: not tested
			# one file = one document
			if self.doc_separator is None:
				d = text
				doc = self.parseDoc(d)
				yield doc
			# one file = several documents
			else:
				if self.stacked_json:
					for line in text:
						doc = self.parseDoc(json.loads(line))
						yield doc
				else:
					for d in text:
						doc = self.parseDoc(d)
						yield doc

class TxtAdapter(FileAdapter):
	"""
	Transform a text file in a set of TextDoc
	"""

	def __init__(self, doc_separator="\n"):
		self.doc_separator = doc_separator

	def readFile(self, fp):
		documents = []
		with open(fp) as f:
			# one file = one document
			if self.doc_separator is None:
				doc = f.read().rstrip("\n")
				doc = TextDoc(doc)
				documents.append(doc)
			# one file = several documents
			else:
				list_docs = f.read().rstrip("\n").split(self.doc_separator)
				list_docs = [TextDoc(text) for text in list_docs]
				documents.extend(list_docs)
		return documents

