
from typing import List

from flashtext.keyword import KeywordProcessor

from collections import Counter

import re

from i3.nlp.textprocessor import TextProcessor

import networkx as nx

###

def buildKeywordProcessor(list_keywords: List[str], replace_fun=lambda x: x):
	""" build a keyword processor with a given list """
	keyword_processor = KeywordProcessor()
	keyword_names = list_keywords
	clean_names = [x.replace(" ", "_") for x in list_keywords]
	for keyword_name, clean_name in zip(keyword_names, clean_names):
		keyword_processor.add_keyword(keyword_name, clean_name)
	return keyword_processor

def buildKeywordProcessorMWE(list_keywords: List[str], replace_fun=lambda x: x.replace(" ", "_")):
	""" build a keyword preocessor specifically for MWE (replace spaces with underscores) """
	return buildKeywordProcessor(list_keywords, replace_fun)

def extractKeywordsFixedSet(text: str, list_keywords=None, keyword_processor=None):
	""" extract all the keyword of the list appearing in the text"""
	if keyword_processor is None:
		keyword_processor = buildKeywordProcessor(list_keywords)
	keywords_found = keyword_processor.extract_keywords(text)
	return keywords_found

###

class NERExtractor:

	"""
	Named Entity Regular Expression (based) Extractor
	"""

	def __init__(self, fp_topwords=None, list_topwords=[], list_allowed_linking_words=[], max_topwords=100, exclude_single_start=True):

		self.re_capitalized = re.compile(r"^([A-Z][^ ]* ?)+$")

		self.exclude_single_start = exclude_single_start

		self.set_exceptions_head = set()

		# original expression for english
		#self.re_mwe_ent = re.compile(r"([A-Z][^ ]+[ ,.]?((of |of the|for |'s |and |al-)[A-Z][^ ]+[ .,]?)?)+")

		# !!! even though it is a regular expression the order of the words in the list matters
		# so to match "X de la Y", "de" should comme after "de la in the list of linking words"

		if not list_allowed_linking_words:
			list_allowed_linking_words = "' |von der |de la |des |di |della |delle |del |dei |degli |el |al |for |'s |and |et |ad-|ad |as |as-|ez-|ez |az-|az |ash-|ash |al-|el-|wa |wa al |wa al-|wa as |wa as-|wa ad |wa ad-|wa az-|wa az |wa ash-|wa ash |y |i |e |-e |ed |ed-|wa ed |wa ed-|a |a-|o |v |& |of the |sur |pour le |pour la |pour les |pour un |pour une |per il |per la |per le |per i |aux |ai |alla |à la |à |von |of |du |de |pour |per ".split("|") # of the, 
#			list_allowed_linking_words += "|" + "|".join([ x.replace(" ", "-") for x in list_allowed_linking_words.split("|")])
			list_allowed_linking_words += [ x.replace(" ", "-") for x in list_allowed_linking_words]

		if False:
			self.set_exceptions_head = set(["New"])

		# does not work, seems due to sentences segmentation TODO check that
		list_allowed_prefix_words = ["pro-", "anti-", "non-", "ex-", "intra-", "inter-", "extra-", "late-", "former-", "proto-", "then-", "ultra-", "all-"]

		# changements:
		# commence par majuscule ou nombre (cardinalité de +)
		# apres majuscule, cardinalité passée de + à *
		# au lieud e mathc (maj puis link puis maj)*, match (maj uis link)* et enlever ensuite potentiél link qui sont en suffixe
		# ajouter bcp bcp de linking words

#		self.re_mwe_ent = re.compile(r"([A-Z][^ ]+[ ,.]?(("+"|".join(list_allowed_linking_words)+r")[A-Z][^ ]+[ .,]?)?)+")
#		self.re_mwe_ent = re.compile(r"(al-|al |as-|as S|ad-|ad |az |az-)?(([A-Z][^ ]+|[0-9])[ ,.	]?(("+"|".join(list_allowed_linking_words)+r")[A-Z][^ ]+[ .,]?)?)+")
		self.re_mwe_ent = re.compile(r"\b(al |al-|as-|ad |ad-|az |az-|el |el-|Dr. )?(([A-Z][^ ]*|[0-9]+)[ ,.	-]?(("+"|".join(list_allowed_linking_words)+r")[ .,-]?)?)+")

# to be used only with non epty allowd prefix list
#		self.re_mwe_ent = re.compile(r"\b(al |al-|as-|ad |ad-|az |az-|el |el-|Dr. "+"|".join(list_allowed_prefix_words)+r")?(([A-Z][^ ]*|[0-9]+)[ ,.	]?(("+"|".join(list_allowed_linking_words)+r")[ .,]?)?)+")

		self.re_trailing_linking = re.compile(r"("+"|".join([" "+x.strip() for x in list_allowed_linking_words])+r")$")

		# load stop words (if specified)

		self.set_topwords = set()

		if fp_topwords:
			with open(fp_topwords) as f:
				for line in f.readlines():
					word, count = line.split()
					if word[0].isupper():
						continue
					list_topwords.append(word)

		if list_topwords:
			self.set_topwords = set(list_topwords)

#	def extract_strictly_capitalized_tokens(self, data):
#		ret = []
#		for p in data:
#			if re.search(self.re_capitalized, p):
#				ret.append(p)
#		return ret

	def process_text(self, d, ignore_first=True):
		return self.extract_MWE_ents([d], ignore_first)

	def extract_MWE_ents(self, data, ignore_first=True):
		ret = []
#		print(">>>>>>", data)
		for p in data:
			p = p.strip()
#			print(p)
			match = re.search(self.re_mwe_ent, p)
			match_dashed = re.search(self.re_mwe_ent, p.replace("-", " "))

			while match:
				# if there is not space but a dash, then tries matching by replaceing dashes with spaces
				# enables to match MWE with linking words such as: Esch-sur-Alzette

				switched_dashed = False
				if " " not in p and "-" in p:
					match = match_dashed
					switched_dashed = True
				ignore=False

#				print(match, match_dashed)
				mwe = match.group(0).strip()

				if not mwe:
					break

				### post processing after matching

				if len(mwe) > 0:
						# remove trailing linking words
						mwe = re.sub(self.re_trailing_linking, "", mwe)

						# remove trailing common word, eg "-held" in "QSD-held"
						mwe = re.sub(r"-[^A-Z][^ -]*$", "", mwe)

				###  filtering after matching

				if False and " " in mwe:
					# if a mwe, discard the first word if its lowered version is a stopword (unless it is part of exceptions)
					head, tail = mwe.split(maxsplit=1)
					if head not in self.set_exceptions_head and head.lower() in self.set_topwords:
						mwe = tail
						mwe = " ".join([x for x in mwe.split(" ") if x != "" and x[0].isupper()])
				else:
					# if begining of sentence and only one word: ignore
					if ignore_first and self.exclude_single_start and (match.start() <= 1 or (match.start() >= 2 and p[match.start()-2:match.start()] == ". ")):
						tokens = mwe.split(" ")
						if len(tokens) == 1:
							ignore =True
						elif not all([len(x) > 0 and x[0].isupper() for x in tokens]):
							ignore = True
					# ignore tokens that are purely numbers with capital letters after
					if len(mwe) > 0 and mwe[0].isdigit() and mwe.count(" ") == 0:
						ignore = True
					# ignore tokens that are one character long
					if len(mwe) == 1:
						ignore = True
					# ignore list of linkings words and numbers
					if mwe.lower() == mwe:
						ignore = True

#				print(">>"+mwe+"<<", ignore, len(mwe) > 0 and mwe[0].isdigit() and mwe.count(" ") == 0)

				# conditionay clean and add the mwe
				if not ignore: # and not mwe.lower() in self.set_topwords:
					ends_in_punct = mwe[-1] == "." or mwe[-1] == "," or mwe[-1] == ")" or mwe[-1] == "'"
					mwe = mwe if not ends_in_punct else mwe[:-1]
#					mwe = re.sub("\w*$", "", mwe)
#					mwe = mwe.strip()

					# when an arabic qualificative adjective with a lunar letter also happens to be an English word,
					# need to check that the next words starts sith the letter is it supposed

					# TODO: 2 it for in case of more than one "as" in text
					if " as " in mwe:
#						part1, part2 = mwe.split(" as ", maxsplit=1)
#						part2h = part2.split()[0]
#						if part2h in self.set_topwords or part2h.lower() in self.set_topwords or (len(part2h) > 1 and part2h[1].isupper()):
#							ret.append(part1)
#							ret.append(part2)
#						else:
#							ret.append(mwe)
						part1, part2 = mwe.split(" as ", maxsplit=1)
						part2h = part2.split()[0]
						if (len(part2h)) >= 2 and part2h[0].lower() == "s" and part2h[1].islower():
								ret.append(mwe)
						else:
							ret.append(part1)
							ret.append(part2)

					elif " ash " in mwe:
						part1, part2 = mwe.split(" ash ", maxsplit=1)
						part2h = part2.split()[0]
						if (len(part2h)) >= 3 and part2h[:2].lower() == "sh" and part2h[2].islower():
							ret.append(mwe)
						else:
							ret.append(part1)
							ret.append(part2)
					else:
						ret.append(mwe)
					

				p = p[match.end():]
				match = re.search(self.re_mwe_ent, p)
				match_dashed = re.search(self.re_mwe_ent, p.replace("-", " "))

		return ret


###

class Rake:

	"""
	RAKE

	uses stopwords to extract keywords and keyphrases
	"""


	def __init__(self, lang, re_stopwords=None, stop_words=None, invert=False):
		self.lang = lang

		if stop_words:
#			print(r"\b("+r"|".join(stop_words)+r")\b")
			self.re_stopwords = re.compile(r"\b("+r"|".join(stop_words)+r")\b")
		if re_stopwords:
			self.re_stopwords = re_stopwords

		self.invert = invert

	def process_text(self, text, return_counter=False):
#		print("TEXT", text)
		segments = TextProcessor.splitText(text, split_words=False)
#		print("SSS", segments)
		return self.process_segments(segments, return_counter)

	def process_segments(self, segments, return_counter=False):
		phrases = TextProcessor.extract_segment_type(segments, "PROP:D")
#		print(phrases)
		return self.process_phrases(phrases, return_counter)

	def process_phrases(self, segments, return_counter=False):

		count = Counter()

		for p in segments:
			p = p.strip()

#			if self.lang == "en":
#				p = p.replace("of the", "of_the")
	
			pieces = self.re_stopwords.split(p)

#			print("PPP", p)
#			print(pieces)

			pieces_tagged = []

			for i in range(len(pieces)-1):
				pieces_tagged.append( [pieces[i] , ("p" if i%2 == (0 if not self.invert else 1) else "_")])
			
			if not self.invert:
				pieces_tagged.append( [pieces[-1], "p"] )

#			print(pieces_tagged)

			for ss, tag in pieces_tagged:
				if tag == "p" and ss.strip():
					ss = ss.strip()
					ss = ss if ss and ss[0] != "'" else ss[1:]
					ss = ss if ss and ss[-1] != "'" else ss[:-1]
					if ss:
#						print(ss)
						count.update([ss])

#		print("CCC", count)

		if return_counter:
			return count

		return count.most_common()


import numpy as np

class GraphBasedKWExtractor:

	def __init__(self, measure="degree", inclusion="subset", threshold=0.5, min_tok=1, stop_words=[]):

		self.measure = measure
		self.inclusion = inclusion
		self.threshold = threshold
		self.min_tok = min_tok
		self.set_stopwords = set(stop_words)

	def process_text(self, text):
#		print("TEXT", text)
		segments = TextProcessor.splitText(text, split_words=False)
		return self.process_segments(segments)

	def process_segments(self, segments):
		phrases = TextProcessor.extract_segment_type(segments, "PROP:D")
		return self.process_phrases(phrases)

	def process_phrases(self, segments):

		import datetime
		import sys

		data = []

		for ph in segments:
			ph = ph.strip()

			len_w = ph.count(" ") + 1

			if len_w >= self.min_tok:
				data.append((ph, len(ph), len_w, None))

#		print("* sorting",  datetime.datetime.now(), file=sys.stderr)

		data.sort(key=lambda x: x[2])

		#print(data)

#		print("* graphing",  datetime.datetime.now(), file=sys.stderr)

		graph = nx.Graph()

		map_node_weights = {}

		for idx_1 in range(len(data)):
#		for idx_1 in range(len(data)-1):
			sent1, len_s1, len_w1, set_words1 = data[idx_1]
			if len_w1 < self.min_tok:
				continue

			if self.inclusion == "subset":
				set_s1 = set(sent1.split()) - self.set_stopwords
			
			if self.inclusion == "subsetpref":
				set_s1 = set(sent1.split()) - self.set_stopwords

				set_s1 = set([x[:round(len(x)*2./3)] if len(x) >= 6 else x for x in set_s1])
			#

			for idx_2 in range(len(data)):
#			for idx_2 in range(idx_1 + 1, len(data)):
				sent2, len_s2, len_w2, set_words2 = data[idx_2]
				#

				if idx_1 == idx_2:
					continue

				if len_w2 < self.min_tok:
					continue

				if len_w2 < len_w1:
					continue

#				if len_s2 < len_s1:
#					continue


				if self.inclusion == "subset":
					set_s2 = set(sent2.split()) - self.set_stopwords

				if self.inclusion == "subsetpref":
					set_s2 = set(sent2.split()) - self.set_stopwords

					set_s2 = set([x[:round(len(x)*2./3)] if len(x) >= 6 else x for x in set_s2])


#				print("*", sent1, "%", sent2)

				add_edge = False

				if self.inclusion == "strict":
					if sent1 in sent2:
						add_edge = True
				elif self.inclusion == "subset":
					isect = (set_s1 & set_s2) #- self.set_stopwords
#					print(len(isect), len(set_s1), len(set_s2))

					if isect and len(isect) >= len(set_s1) * self.threshold :
						add_edge = True
				elif self.inclusion == "subsetpref":
					isect = (set_s1 & set_s2) #- self.set_stopwords
#					print(len(isect), len(set_s1), len(set_s2))

					if isect and len(isect) >= len(set_s1) * self.threshold :
						add_edge = True

				else:
					raise Exception("invalid inclusion method", self.inclusion)

#				print(add_edge, "*", "    ", isect, len(isect), len(set_s1), len(set_s2), "\n   ", sent1, "\n   ", sent2)

				if add_edge:
					w = 0
					if graph.has_edge(sent1, sent2):
						w = graph[sent1][sent2]["weight"]
					graph.add_edge(sent1, sent2, weight=w+1)

					if sent1 not in map_node_weights:
						map_node_weights[sent1] = 0
					
					if self.inclusion == "subset":
#						score = 1
						score = (len_w2 / len(isect)) * np.log(len_s1)
#						score = len(isect) / len_w2
						map_node_weights[sent1] += score if score > 0.01 else 0.01
#						map_node_weights[sent1] += len_w2 / len(isect) if len_w2 > 0 else 0
#						map_node_weights[sent1] += len(isect) / len_w2 if len_w2 > 0 else 1
#						map_node_weights[sent1] += len(isect) / len_w1 if len_w1 > 0 else 0

					if self.inclusion == "strict":
						map_node_weights[sent1] += 0

#		print("* writting",  datetime.datetime.now(), file=sys.stderr)

		nx.write_gexf(graph, "subsum.gexf")
		nx.write_edgelist(graph, "subsum.edges", delimiter="\t")

#		print(map_node_weights)


		if graph.number_of_nodes() == 0:
			return []

		nodes_items = None
		
		if self.measure == "degree":
			nodes_items = list(graph.degree)
#			nodes_items = [ [x[0], round( x[1] / len(x[0]) , 2) ] for x in graph.degree]

		elif self.measure == "centrality":

			try:
				centrality = nx.eigenvector_centrality_numpy(graph)
			except Exception as e:
				print(e)
				return []

#			nodes_items = [ [v,c * graph.degree[v]] for v, c in centrality.items() ]
			nodes_items = [ [v,c] for v, c in centrality.items() ]

		elif self.measure == "clustering":

			clustering = nx.clustering(graph)

#			print(clustering)

			nodes_items = [ [v,c] for v, c in clustering.items() ]


		elif self.measure == "centr+clust":

			try:
				centrality = nx.eigenvector_centrality_numpy(graph)
			except Exception as e:
				print(e)
				return []

			clustering = nx.clustering(graph)

#			print(clustering)

			nodes_items = [ [v, c * centrality[v]] for v, c in clustering.items() ]



		else:
			raise Exception("invalid measure:", self.measure)


#		nodes_items = [ [x[0], round( x[1] / len(x[0]) , 2) ] for x in graph.degree]
#		nodes_items = [ [x[0], round( x[1] / len(x[0]) * map_node_weights[x[0]] , 2) ] for x in graph.degree]


#			nodes_items = [ [v,c * graph.degree[v]] for v, c in centrality.items() ]

		keywords = sorted(nodes_items, key=lambda x: x[1], reverse=True)

#		print("KBKW", keywords)

		if not keywords:
			return []

		median = int(round(len(keywords)/2.)) if len(keywords) > 1 else 0

		median_degree = keywords[median][1]

#		keywords = [x for x in keywords if x[1] >= median_degree]


#		print(keywords)

		return keywords

import configparser

class CategoryConverter:

	"""
	Class to convert list of EMM categories to list of labels

	the convertion is specified in an .ini file
	in following format:

		[categories]
		categ_name1 = keyword1,keywword2
		categ_name2= keyword3,keyword4,keyword5
		...
	"""

	def __init__(self, conf_fp):

		categ_conf = configparser.ConfigParser()
		categ_conf.read(conf_fp)

		if "categories" not in categ_conf:
			raise Exception("error in configuration file" + conf_fp)

		#

		self.list_labels = list(categ_conf["categories"].keys())
		self.set_labels = set(categ_conf["categories"].keys())

		self.map_emmcateg_label = {}

		for label, list_emmcateg in categ_conf["categories"].items():
			for categ in list_emmcateg.split(","):
				self.map_emmcateg_label[categ] = label

		#print(self.map_emmcateg_label)

	def convertCategoryToLabel(self, list_categ, absent_strategy="error", uniq=False):
		""" convert a list of emm categories to a list of labels
		list_categ: list of elements to convert
		absent_strategy: behaviour when element is no in map table (error, skip, include, None)
		uniq: can elements be repeated"""
		res = []
		for categ in list_categ:
			if categ in self.map_emmcateg_label:
				label = self.map_emmcateg_label[categ]
				res.append(label)
			else:
				print("Warning: key absent from table:" + str(categ))

				if absent_strategy == "skip":
					continue
				elif absent_strategy == "include":
					res.append(categ)
				elif absent_strategy == "None":
					res.append("")
				elif True or absent_strategy == "error":
					raise Exception("key absent from table:" + str(categ))
		
		res = res if not uniq else list(set(res))
		res = [x.upper() for x in res if x is not None]

		return res


if __name__ == "__main__":

	re_stopwords_en = re.compile(
		r"\b([Tt]he|of|[Ii]n|is|are|not|to|have|has|ve|[Aa]nd|or|[Oo]n|[Ff]or|his|her|their|its|before|after|" +
		r"[Aa]n?|[Tt]his|[Tt]hat|should|must|would|[Aa]s|was|were|also|when|[Tt]h[oe]se|though|" +
		r"[Bb]y|[Ww]ho|with|without|from|will|may|be|been|had|might|could|[Aa]t|than|[Ss]ince|[Nn]either|" +
		r"nor|our|[Ww]hile|[Tt]hese|[Ww]here|into|[Ii]f|[Dd]o|[Dd]id|how|[Ww]hat|[Ww]hen|within|such|both|each|"+
		r"[Ww]hether|amid|[Aa]mong|can|which|thus|therefore|doesn't|didn't|[Nn]otwithstanding|"+
		r"[Tt]here|own|[Tt]heir|[Bb]ut|[Hh]owever|[Oo]nly|" +
		r"against|upon|still|"+
		r"your|[Hh]is|[Hh]er|'s|'|being|but|[Ff]or|m|am|d|t|re)\b")


	re_stopwords_fr = re.compile(

		r"\b("+
	#	r"\b([Ll][ae']|[Ll]es|[Cc]es?|[Cc]ette|[Qq]u[ei']|[Uu]ne?|[Qq]uels?|[Ll]es?quels?"+
		r"à un tel point que|à tel point que|à ce que|à condition que|à mesure que|à moins que|à seule fin que|afin de|afin que|ainsi que|alors que|après que|attendu que|au cas où|au fur et à mesure que|au point que|aussi bien que|aussitôt que|autant que|avant que|bien que|cependant que|c'est pourquoi|comme|comme quoi|comme si|dans la mesure où|d'autant plus que|d'autant que|de manière que|de manière à ce que|de manière à|de sorte que|de sorte à|de sorte à ce que|de telle manière que|de telle sorte que|de crainte que|de façon que|de même que|de peur que|depuis que|dès lors que|dès que|d'ici que|du fait que|du moment que|durant que|étant donné que|en attendant que|en cas que|en sorte que|encore que|jusqu'à ce que|loin que|lors même que|même si|mis à part le fait que|où que|parce que|pendant que|pour peu que|pour que|pourvu que|puisque|quand bien même que|quoique|qui que|sans que|selon que|sauf que|si bien que|si ce n'est que|sitôt que|suivant que|tandis que|tant que|tellement que|une fois que|vu que|pour|lorsque|puisque|quand|que|qui|qu'|" +
		r"[Pp]our|[Ll][ae]|[Ll]es|[Dd]es|[Dd][ue]|[Cc]es?|[Cc]ette|[Qq]u[ei']|[Uu]ne?|[Qq]uels?|"+
		r"[Pp]ar|et|ou|[àÀ]|[Aa]ux?|[Ll]'?|d'|[Ee]n|avec|[Mm]es|[Mm]on|[Mm]a|[Ss]a|[Ss]es|[Ll]eurs?|"+
		r"afin de|afin que|de sorte à|en fonction d[ue]|[Dd]ans|est|sont|ne sont pas|a|ont|n'a pas|n'ont pas"+
	#	r"chez|sur|sous"+
		r")\b")



	print("* extract keywords")

	text = "the quick brown fox jumps over the lazy cat"
	
	list_kw = list(["brown fox", "lazy cat", "jump", "jumps"])

	print(extractKeywordsFixedSet(text, list_kw))

	text = "Testing how good Named Entity Recognition works. When using Regular Expressions, the Brown Fox and the Lazy Cat met the President of the United Swamps."

	print("* NERExtractor")

	nerext = NERExtractor(list_allowed_linking_words=[])

	print(nerext.extract_MWE_ents([text]))

	nerext = NERExtractor(list_allowed_linking_words=["and ", "of ", "of the "])

	print(nerext.extract_MWE_ents([text]))

	print("* RAKE")

	rake = Rake("en", re_stopwords_en)

	list_kw_rake = rake.process_text(text)
	print(list_kw_rake)

	text = "By the signature of this Treaty, the involved parties give proof of their determination to create the first supranational institution and that thus they are laying the true foundation of an organised Europe. This Europe remains open to all European countries that have freedom of choice. We profoundly hope that other countries will join us in our common endeavour."

	text = """All human beings are born free and equal in dignity and rights. They are
endowed with reason and conscience and should act towards one another in a
spirit of brotherhood."""

	list_kw_rake = rake.process_text(text)
	print(list_kw_rake)

	rake = Rake("en", re_stopwords_en, invert=True)

	list_kw_rake = rake.process_text(text)
	print(list_kw_rake)

	text = """Tous les êtres humains naissent libres et égaux en dignité et en droits. Ils sont
doués de raison et de conscience et doivent agir les uns envers les autres dans un
esprit de fraternité."""

	rake = Rake("fr", re_stopwords_fr)

	list_kw_rake = rake.process_text(text)
	print(list_kw_rake)
