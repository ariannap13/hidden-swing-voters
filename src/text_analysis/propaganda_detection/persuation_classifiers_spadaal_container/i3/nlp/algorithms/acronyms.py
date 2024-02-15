
import sys
import re
from collections import Counter

from .algorithm import CorpusAlgorithm

class AcronymExtractor(CorpusAlgorithm):

	def __init__(self, lower_all=True):

		self.re_acronym = re.compile(r" \([^-: ]+\)[.,:; ]")
		self.lower_all = lower_all


	def processCorpus(self, corpus):
		
		all_res = []
		
#		for doc in corpus.getDocuments():
#			res = self.processDocument(doc.text)
#			print(res)
#			all_res.append(res)

		all_res = corpus.applyTransformationOnCorpus(self.processDocument, replace=False, process_tokens=False)

		map_abrev_mwe = Counter()
		for res in all_res:
			for abrev, counter_mwe in res.items():
				if abrev not in map_abrev_mwe:
					map_abrev_mwe[abrev] = Counter()
				map_abrev_mwe[abrev].update(counter_mwe)

		return map_abrev_mwe

	def processDocument(self, text):
		""" search for abreviation and the corresponding MWE acronym """

		map_abrev_mwe = {}

		for match in re.finditer(self.re_acronym, text):

			acronym_orig = match[0][2:-2]

			acronym = acronym_orig

			if acronym.isdecimal():
				continue
			
			if len(acronym) < 2:
				continue

			subline = text[:match.start()]
			
			words = subline.replace("-", " ").replace("'", " ").split(" ")

			words_orig = words

			if self.lower_all:
				acronym_lower = acronym.lower()

				words_lower = [x.lower() for x in words]

				words = words_lower
				acronym = acronym_lower

			words = words[::-1]
			acronym = acronym[::-1]

			count=0
			for letter in acronym:
				if count == len(words):
					break
				if len(words[count]) > 0 and words[count][0] == letter:
					count += 1
					continue
				elif count+1 < len(words) and len(words[count+1]) > 0 and words[count+1][0] == letter:
					count += 2
					continue
				elif count+2 < len(words) and len(words[count+2]) > 0 and  words[count+2][0] == letter:
					count += 3
					continue
				else:
					break

			found = count == len(acronym)
			if found:

				mwe = " ".join(words_orig[-count:])

				# if there is punctuation, discard the MWE (avoid  roman numeral: (i), (ii) (iii), (iv), etc.)
				if re.search(r"[,;.]", mwe):
					continue 

				acronym = acronym_orig

				if acronym not in map_abrev_mwe:
					map_abrev_mwe[acronym] = Counter()
				map_abrev_mwe[acronym].update([mwe])

		return map_abrev_mwe
