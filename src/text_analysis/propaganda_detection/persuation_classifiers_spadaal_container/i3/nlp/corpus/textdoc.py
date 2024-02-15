#from __future__ import annotations

from i3.nlp.textprocessor import TextProcessor

class TextDoc:

	def __init__(self, text, tags={}, tokenize=False):
		self.text = text
		self.tokens = [] if not tokenize else text.split()
		self.tags = tags


	def __repr__(self):
		ret = "TextDoc("+ \
			(((str(self.text[:50] + " ... " + self.text[-50:]) if len(self.text) > 100 else self.text ) if self.text is not None else "")) + \
			"," + \
			((str(self.tokens[:10]) + " ... " + str(self.tokens[-10:])) if len(self.tokens) > 50 else str(self.tokens)) + \
			"," + \
			str(self.tags) + \
			")"
		return ret

	def toCsv(self, include_header=True, sep=",", replace_newline="   ", replace_sep="  "):
		ret = ""
		if include_header:
			ret += sep.join(list(self.tags.keys()))
			ret += sep+"text\n"
		data = list(self.tags.values())
		data = [x.replace("\n", replace_newline) for x in data]
		if replace_sep:
			data = [x.replace("\t", replace_sep) for x in data]
		ret += sep.join(data)
		ret += sep+self.text.replace("\n", replace_newline)
		ret += "\n"
		return ret

	def tokenize(self, tokenizer="split"):
		self.tokens = TextProcessor.splitWords(self.text, tokenizer=tokenizer)


	@staticmethod
#	def preprocess(doc: TextDoc, tokenizer="split", replace_nonwords=False) -> TextDoc:
	def preprocess(doc, tokenizer="split", replace_nonwords=False):
		# preprocessing on string
		TextDoc.applyTransformationOnDocument(doc, fun=TextProcessor.removeHyphens, process_tokens=False)
		TextDoc.applyTransformationOnDocument(doc, fun=TextProcessor.removeHtml, process_tokens=False)
		TextDoc.applyTransformationOnDocument(doc, fun=TextProcessor.normalizeCharacters, process_tokens=False)
		if replace_nonwords:
			TextDoc.applyTransformationOnDocument(doc, fun=TextProcessor.replaceNonWordsWithSpecialTokens, process_tokens=False)
		# preprocessing on tokens
		doc.tokenize(tokenizer) 		#doc.tokens = TextProcessor.splitWords(doc, tokenizer=tokenizer)
		TextDoc.applyTransformationOnDocument(doc, fun=TextProcessor.stripPunctuation, process_tokens=True, process_tokens_list=False)
		TextDoc.applyTransformationOnDocument(doc, fun=TextProcessor.filterStopWords, process_tokens=True	)
		return doc

	@staticmethod
	def applyTransformationOnDocument(doc, fun=lambda x: x, replace=True, process_tokens=True, process_tokens_list=True, process_doc=False, add_field=None):
		""" apply function on a single document
		
		fun: function to apply
		replace: in place replace
		process_tokens: if false process the original text, if true the tokens
		process_tokens_list: if trues, process all tokens as a single object, if false process each token individually
		add_filed: add a new field and initialiaze it with the computed value
		log: whether to log progress on stdout
		"""

		if process_doc:
		# process document as a whole
			res = fun(doc)
			if replace:
				return res
			elif add_field is not None:
				setattr(doc, add_field, res)
				return None
			else:
				return res
		# process splited tokens, do not touch text
		elif process_tokens:
			if process_tokens_list:
				# process token list as a single object
				res = fun(doc.tokens)
			else:
				# process each token individually
				res = [fun(x) for x in doc.tokens]
			if replace:
				doc.tokens = res
				return None
			elif add_field is not None:
				setattr(doc, add_field, res)
				return None
			else:
				return res
		# process text, do not touch splited tokens
		else:
			res = fun(doc.text)
			if replace:
				doc.text = res
				return None
			elif add_field is not None:
				setattr(doc, add_field, res)
			else:
				return res

if __name__ == '__main__':
	pass