

import unicodedata

import re

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

from collections import namedtuple

Segment = namedtuple("Segment", ["type", "text", "children", "start", "end", "offset", "tags"])


RE_ZWSP = re.compile(r"[\u200B\u2060\uFEFF]+")
RE_NONBREAKING_SPACE = re.compile(r"[^\S\n\v]+", flags=re.UNICODE)

class TextProcessor:

	translation_table = None

	def __init__(self):
		pass

	# split text -> tokens

	@staticmethod
	def splitParagraphs(text, sep="\n"):
		return text.split(sep)

	@staticmethod
	def splitSentences(text, tokenizer="nltk"):
		if tokenizer == "nltk":
			return sent_tokenize(text)
		elif tokenizer == "i3":
			segments = TextProcessor.splitText(text)
			sentences = TextProcessor.extract_segment_type(segments, "SENT:D")
			return sentences

	@staticmethod
	def splitWords(text, tokenizer="nltk"):
		if tokenizer == "nltk":
			return word_tokenize(text)
		if tokenizer == "split":
			return text.split(" ")
		elif tokenizer == "i3":
			segments = TextProcessor.splitText(text)
			sentences = TextProcessor.extract_segment_type(segments, "WORD:D")
			return sentences

	@staticmethod
	def replaceNonBoundaryDotsWithToken(text):
		# this function and the reverse one are ad-hoc function used when
		# splitting a text around punctuation, while not losing punctuation
		# character who semantically do not represent a punctuation
		# e.g. in floating numbers, in abreviation, after German numbers


		list_dots_to_ignore = []

		def find_previous_word(text, from_pos):
			i = from_pos - 1
			while i >= 0 and text[i] != " ":
				i = i-1
			return text[i+1:from_pos]

		def find_next_word(text, from_pos):
			i = from_pos + 1
			while i < len(text) and text[i] != " ":
				i = i+1
			return text[i+1:from_pos+1]

		for i, c in enumerate(text):
			if c == ".":
				prev_word = find_previous_word(text, i)

				stat_words = [ [x.islower(), x.isupper(), x == "."] for x in prev_word]
				
				is_initial = sum([x[1] for x in stat_words]) == sum([x[2] for x in stat_words])+1 and sum([x[0] for x in stat_words]) == 0

				#is_number = prev_word.isnumeric() and i+2 < len(text) and text[i+1].isnumeric()
				is_number = prev_word.replace("-", "").replace("(", "").isnumeric() and i+2 < len(text) and text[i+1].isnumeric()

				abrev_2 = text[i-1:i+3]

				is_abrev2 = len(abrev_2) >= 4 and abrev_2[-1] == "." # and (abrev_2 in "i.e." or abrev_2 in "e.g.")

				is_title = prev_word in ["Dr", "Pr", "Prof", "Gov", "St"]

				if is_abrev2:
					list_dots_to_ignore.append(i)				
					list_dots_to_ignore.append(i+2)
					i+=1
					continue

				if is_initial or is_number or is_title:
					list_dots_to_ignore.append(i)


		for i in sorted(list_dots_to_ignore, reverse=True):
			text = text[:i] + "@dot@" + text[i+1:]

		return text

	@staticmethod
	def rootSegment(text, keep_indices=False):
		start = 0 if keep_indices else None
		end = len(text) if keep_indices else None 
		offset = 0 if keep_indices else None
		tags = {}
		return Segment(*["TEXT", text, [], start, end, offset, tags]) if type(text) == str else text

	@staticmethod
	def removeSpecialDotToken(x):
		return x.replace("@dot@", ".").replace("dot@", ".")

	@staticmethod
	def process_segment(data, regexp, tag, drop_orig: bool=True, preproc_fun=lambda x: x, postproc_fun=lambda x: x, keep_indices=False):


		segment_type, segment_text, subsegment_pieces = data.type, data.text, data.children

		# if no subsegment, recursive call
		if len(subsegment_pieces) != 0:
			for subseg in subsegment_pieces:
				TextProcessor.process_segment(subseg, regexp, tag, drop_orig, preproc_fun, postproc_fun, keep_indices)
		else:

			# process only data segments
			if segment_type[-2:] == ":D" or segment_type == "TEXT":
				# apply preprocessing to text
				segment_text = preproc_fun(segment_text)

				# split according to re

				if not keep_indices:
					segment_pieces = re.split(regexp, segment_text)

					# create sub segments
					segment_tokens = [ Segment(*[tag+":D", postproc_fun(segment_pieces[i]), [], None, None, None, {}]) if i % 2 == 0 else Segment(*[tag+":S", segment_pieces[i], [], None, None, None, {}]) for i in range(len(segment_pieces)) ]
					subsegment_pieces.extend(segment_tokens)

				else:
					segment_pieces = []

					list_matches = list(re.finditer(regexp, segment_text))
					list_idx = sorted([x.start() for x in list_matches] + [x.end() for x in list_matches])

					# if first and last index not in list, add them (deals with no match, while avoiding duplicates)
					if 0 not in list_idx[:1]:
						list_idx = [0] + list_idx
					if len(segment_text) not in list_idx[-1:]:
						list_idx = list_idx + [len(segment_text)]
						
					segment_pieces = [ [segment_text[list_idx[i]:list_idx[i+1]], list_idx[i], list_idx[i+1]] for i in range(len(list_idx)-1)]

					# create sub segments

					# the fast way:
					#segment_tokens = [ [tag+":D", postproc_fun(segment_pieces[i][0]), [], segment_pieces[1], segment_pieces[2]] if i % 2 == 0 else [tag+":S", segment_pieces[i][0], []] for i in range(len(segment_pieces)) ]
					#subsegment_pieces.extend(segment_tokens)

					# the understable way:
					segment_tokens = []
					for i in range(len(segment_pieces)):
						idx_start = segment_pieces[i][1]
						idx_end = segment_pieces[i][2]
						offset = 0 if data.offset is None else data.offset + data.start
						tags = {}
						if i % 2 == 0:
							piece = Segment(*[tag+":D", postproc_fun(segment_pieces[i][0]), [], idx_start, idx_end, offset, tags])
						else:
							piece = Segment(*[tag+":S", segment_pieces[i][0], [], idx_start, idx_end, offset, tags])
						#print(i, piece)
						subsegment_pieces.append(piece)

				ret = segment_type, segment_text if not drop_orig else "", subsegment_pieces
				return ret

			# otherwise return segment as such
			else:
				return data


	@staticmethod
	def splitTextParagraphs(text, keep_indices=False):
		# 2 or more consecutives new line characters
		analyzed_text = TextProcessor.rootSegment(text, keep_indices) if type(text) == str else text
		TextProcessor.process_segment(analyzed_text, r'(\n\n\n*)', "PAR", keep_indices=keep_indices)
		return analyzed_text

	@staticmethod
	def splitTextLevels(text, keep_indices=False):
		# quotations: "
		# parenthesis: ()
		# square brackets: []
		analyzed_text = TextProcessor.rootSegment(text, keep_indices) if type(text) == str else text
		TextProcessor.process_segment(analyzed_text, r'("|\]|\[|\(|\))', "LVL", keep_indices=keep_indices)
		return analyzed_text

	@staticmethod
	def splitTextSentences(text, keep_indices=False):

		analyzed_text = TextProcessor.rootSegment(text, keep_indices) if type(text) == str else text

		# todo: add Arabic, CJK and Devanagari punctuation marks
		punct_chars = r"[\.|;|·|!|\?|:|…|•|\n]"

		TextProcessor.process_segment(
			analyzed_text,
			r'('+punct_chars+r'+ *\n?|'+punct_chars+r'+\n?$)',
			"SENT",
			preproc_fun=TextProcessor.replaceNonBoundaryDotsWithToken,
			postproc_fun=TextProcessor.removeSpecialDotToken,
			keep_indices=keep_indices
		)

		return analyzed_text

	@staticmethod
	def splitTextPropositions(text, keep_indices=False, split_levels=False):
		analyzed_text = TextProcessor.rootSegment(text, keep_indices) if type(text) == str else text

		# removed "-" from regexp in order to match city names likes "New-York" and prevent spliting the mwe in two propositions

#		TextProcessor.process_segment(analyzed_text, r"( *[,–\-:\." + (r'()"' if split_levels else "") + r"]  *|\.$|,$|!$|\?$|^-)", "PROP", keep_indices=keep_indices)
		TextProcessor.process_segment(analyzed_text, r"( *[,–:\.()\"] *|\.$|,$|!$|\?$|^-)", "PROP", keep_indices=keep_indices)



		return analyzed_text

	@staticmethod
	def splitTextWords(text, keep_indices=False):
		analyzed_text = TextProcessor.rootSegment(text, keep_indices) if type(text) == str else text

		TextProcessor.process_segment(analyzed_text, r"(  *|/)", "WORD", keep_indices=keep_indices)

		return analyzed_text


	@staticmethod
	def splitText(text, split_paragraph=True, split_levels=True, split_sentences=True, split_propositions=True, split_words=True, keep_indices=False):
		res = text
		if split_paragraph:
			res = TextProcessor.splitTextParagraphs(res, keep_indices=keep_indices)
		if split_levels:
			res = TextProcessor.splitTextLevels(res, keep_indices=keep_indices)
		if split_sentences:
			res = TextProcessor.splitTextSentences(res, keep_indices=keep_indices)
		if split_propositions:
			res = TextProcessor.splitTextPropositions(res, keep_indices=keep_indices, split_levels=not split_levels)
		if split_words:
			res = TextProcessor.splitTextWords(res, keep_indices=keep_indices)
		return res

	@staticmethod
	def display_segment(segment, lvl=0):

		segment_type, segment_text, subsegment_pieces = segment

		print("  "*lvl, segment_type)
		print("  "*lvl, ">>" if segment_type[-1] != "S" else "__", segment_text, "<<" if segment_type[-1] != "S" else "__")
		for ss in subsegment_pieces:
			TextProcessor.display_segment(ss, lvl+1)

	@staticmethod
	def merge_segment(segment):
		return "".join(TextProcessor.extract_segment_type(segment, "", match_any=True))

	@staticmethod
	def extract_segment_type(segment, type_, match_any=False, return_segment=False):

		segment_type, segment_text, subsegment_pieces = segment.type, segment.text, segment.children

		ret = []
		
		if match_any or segment_type == type_:
			if segment_text:
				ret.append(segment_text if not return_segment else segment)
		else:
			for ss in subsegment_pieces:
				ret.extend(TextProcessor.extract_segment_type(ss, type_, match_any, return_segment))

		return ret

	@staticmethod
	def extract_segment_words(segment):
		return TextProcessor.extract_segment_type(segment, "WORD:D")

	@staticmethod
	def extract_segment_pos(segment, position, match="*"):
		""" extract segment at given position, if match is not specified, return the deepest possible segment """

		#print("ESP", segment.type, segment.start, segment.end, segment.offset, position, match)
		if segment.start is not None:
			offset = segment.offset
			if offset + segment.start <= position and position <= offset + segment.end:
				if (not segment.children and match == "*") or match == segment.type:
					return segment
				else:
					for subsegment in segment.children:
						ret = TextProcessor.extract_segment_pos(subsegment, position, match)
						if ret is not None:
							return ret
		else:
			return None

		return None

	@staticmethod
	def insert_segment_tag_pos(segment, position, key, value, match="*", append=False):
		""" extract segment at given position, if match is not specified, return the deepest possible segment """

		#print("ISP", segment.type, segment.start, segment.end, segment.offset, position, match)
		if segment.start is not None:
			offset = segment.offset
			if offset + segment.start <= position and position <= offset + segment.end:
				if (not segment.children and match == "*") or match == segment.type:
					if segment.tags is None:
						segment.tags = {}
					if append:
						if key not in segment.tags:
							segment.tags[key] = []
						segment.tags[key].append(value)
					else:
						segment.tags[key] = value
					return segment
				else:
					for subsegment in segment.children:
						ret = TextProcessor.insert_segment_tag_pos(subsegment, position, key, value, match, append)
						if ret is not None:
							return ret
		else:
			return None

		return None

#	@staticmethod
#	def splitText(text):
#
#		# not end of sentences
#		# Dr. Aaa
#		# the 13. 
#		
#		# end of sentences
#		# ". "
#
#		# end of proposition
#		# ", "
#
#		#text = TextProcessor.proccessNonBoundaryDots(text)
#
#		analyzed_text = [ text, []]
#
#		# split into paragraphs
#		TextProcessor.process_segment(analyzed_text, r'\n\n\n*')
#
#		# todo: handle ...
#
#		# split into sentences
#		TextProcessor.process_segment(analyzed_text, r'[.!?:…]( |$)',
#			preproc_fun=TextProcessor.replaceNonBoundaryDotsWithToken,
#			postproc_fun=TextProcessor.removeSpecialDotToken)
#	
#		# TODO: handle bug: "from 32 percent in 2003. Finally"  "Figure 2. Revised elephant"
#		# TODO: fix bug " other nontraditional workers To reach more prospective workers"
#		# TODO: handle this? "ammunition.In"
#		# TODO: handle this? "terrorists' targeting"
#		# Daesh terrorists --
#		# terrorist related,' says police department
#		# aiding 'terror' group
#		# Bangkok - Eighteen Thai fishermen rescued
#		# Ukrainian nationalists' organization
#
#		# split quotations apparts
#		TextProcessor.process_segment(analyzed_text, r'[ .,]?" ?')
#
#		# todo: deal with () and []
#		TextProcessor.process_segment(analyzed_text, r"[()]")
#
#		# sentences into propositions
#		TextProcessor.process_segment(analyzed_text, r"[,;–]( |$)")
#		
#		# key phrase extraction
#		#TextProcessor.process_segment(analyzed_text, r" (as well as|and) ")
#
#		return analyzed_text

	# join tokens -> text

	@staticmethod
	def joinParagraphs(tokens):
		return "\n".join(tokens)

	@staticmethod
	def joinSentences(tokens):
		return ". ".join(tokens)

	@staticmethod
	def joinWords(tokens):
		return " ".join(tokens)

	# filter tokens

	### operate on list of strings

	@staticmethod
	def filterStopWords(tokens, word_list=None):
		""" filter stop-words """
		if word_list is None:
			stop_words = stopwords.words('english')
		
			stop_words_bonus = ["The", "December", "January", "April", "In", "I", "II", "III", "IV", "V", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Feb", "Febr", "February", "Mar", "March", "2019", "2020"]

			stop_words.extend(stop_words_bonus)
		else:
			stop_words = word_list

		#words = [ w for w in tokens if w.isalpha()]
		#words = [ w for w in tokens if w not in stop_words and w.lower() not in stop_words and w.upper() not in stop_words]
		words = [ w for w in tokens if w not in stop_words]

		return words

	@staticmethod
	def filterWordsByCount(tokens, count, min_count=None, max_count=None):
		""" filter word by min/max count """

		# TODO: how to deal with OOV ?

		res = [ w for w in tokens if
			(w in count and
				(
					(True if (min_count is None) else count[w] >= min_count)
					and
					(True if (max_count is None) else count[w] <= max_count)
				)
			)]

		return res

	@staticmethod
	def transformWordsStripPunctuation(tokens):
		return [TextProcessor.stripPunctuation(tok) for tok in tokens]

	### operate on strings

	@staticmethod
	def normalizeSpacing(tok):
		""" remove tabs, double spaces, double empty lines, trailing spaces; normalize spacing characters """
		ret = tok
		ret = RE_ZWSP.sub(" ", ret)
		ret = RE_NONBREAKING_SPACE.sub(" ", ret)
		ret = re.sub(u"\x9F", "", ret) # APPLICATION PROGRAM COMMAND
		ret = re.sub("\u2028", "\n\n", ret) # LINE SEPARATOR
		ret = re.sub("[\t\r\f\v]", " ", ret)
		ret = re.sub(" +", " ", ret)
		ret = re.sub(" \n", "\n", ret)
		ret = re.sub("\n ", "\n", ret)
		ret = re.sub("\n\n\n*", "\n\n", ret)
		ret = ret.strip()
		return ret

	@staticmethod
	def normalizeEncoding(tok):
		""" normalize reprsentation of unicode strings to get uniq variant for each displayed character """
		return unicodedata.normalize('NFC', tok)

	@staticmethod
	def normalizePunctuation(tok, char_map=None):
		""" uniformize different stylistic variant of characters """

		# compile table upon first call
		if TextProcessor.translation_table is None:

			default_char_map = {
				# apostrophes
				"‘": "'", # LEFT SINGLE QUOTATION MARK
				"’" : "'", # RIGHT SINGLE QUOTATION MARK
				"＇" : "'", # FULLWIDTH APOSTROPHE
				"‵" : "'", # REVERSED PRIME
				"`" : "'", # GRAVE ACCENT
				"´" : "'", # ACUTE ACCENT
				"′" : "'", # PRIME
				"'" : "'", # APOSTROPHE
				# quotation marks
				"“" : '"', # LEFT DOUBLE QUOTATION MARK
				"”" : '"', # RIGHT DOUBLE QUOTATION MARK
				"„" : '"', # DOUBLE LOW-9 QUOTATION MARK
				"‟" : '"', # DOUBLE HIGH-REVERSED-9 QUOTATION MARK
				"″" : '"', # DOUBLE PRIME
				"˝" : '"', # DOUBLE ACUTE ACCENT
				"＂" :'"', # FULLWIDTH QUOTATION MARK
				"«" :'"', # LEFT-POINTING DOUBLE ANGLE QUOTATION MARK
				"»" : '"',  # RIGHT-POINTING DOUBLE ANGLE QUOTATION MARK
	#			# spaces
	#			u'\xa0' : " ", # NO-BREAK SPACE
	#			'\u200b' : " ", # ZERO WIDTH SPACE
				# dashes
				"–" : "-", # EN DASH
				"—" : "-", # EM DASH
				"﹘" : "-", # SMALL EM DASH
				"⸺" : "-", # TWO-EM DASH
				"⸻" : "-", # THREE-EM DASH
			}

			char_map = default_char_map if char_map is None else char_map

			replace_string = zip(*char_map.items())
			replace_string = list(replace_string)
			translate_from = "".join(replace_string[0])
			translate_to = "".join(replace_string[1])

			TextProcessor.translation_table = str.maketrans(translate_from, translate_to)

		# elipsis
		tok = re.sub("\.\.\.?", "…", tok)

		# double quotes

		#map_mutli = {
		#	"''" : "\"", # (two APOSTROPHE)
		#	"´´" : "\"", # (two ACUTE ACCENT)
		#	"``" : "\"" # (two GRAVE ACCENT)
		#}
		#tok = re.sub("("+("|".join(map_mutli.keys())+")", "\"", tok)

		tok = re.sub("(''|´´|``)", "\"", tok)

		# all other cases
		return tok.translate(TextProcessor.translation_table)

	@staticmethod
	def normalizeCharacters(tok):
		""" applies all the character normalisation functions """
		tok = TextProcessor.normalizeEncoding(tok)
		tok = TextProcessor.normalizePunctuation(tok)
		tok = TextProcessor.normalizeSpacing(tok)
		return tok

	@staticmethod
	def removeHtml(tok):
		""" remove html tags if necessary """
		if re.search("(</|/>)", tok) is not None:
			soup = BeautifulSoup(tok, features="lxml")
			text = soup.get_text()
			return text
		return str(tok)

	@staticmethod
	def removeHyphens(tok):
		""" remove hyphens (only support soft hypens for the moment) """
		# TODO: handle hyphenated words
		text = tok
		text = text.replace("-\n", "")
		text = text.replace("\u00AD\n", "") # soft-hyphen + newline
		text = text.replace("\u00AD", "") # only soft-hyphen
		return text

	@staticmethod
	def stripPunctuation(tok):
		""" remove prefix and postfix punctuation characters from token """

		start=0
		end=len(tok)
		# remove prefix punctuation
		i=0
		while i < len(tok):
			c = tok[i]
			category = unicodedata.category(c)
			if category[0] == "P":
				i += 1
				start = i
			else:
				break

		# remove postfix punctuation
		i=1
		while len(tok) - i > 0:
			c = tok[len(tok)-i]
			category = unicodedata.category(c)
			if category[0] == "P":
				end = len(tok) - i	
				i += 1
			else:
				break

		return tok[start:end]
		
	@staticmethod
	def stripPunctuationOld(tok):
		""" remove prefix and postfix punctuation characterse from token """

		# remove prefix punctuation
		while len(tok) > 0:
			c = tok[0]
			category = unicodedata.category(c)
			if category[0] == "P":
				break
			else:
				tok = tok[1:]

		# remove postfix punctuation
		while len(tok) > 0:
			c = tok[-1]
			category = unicodedata.category(c)
			if category[0] == "P":
				break
			else:
				tok = tok[:-1]

		return tok

	@staticmethod
	def replaceNonWordsWithSpecialTokens(tok, replace_with=" "):
		""" replace urls, email and numbers with special tokens """

		# here to avoid imort conflict on some systems when not striclty needed
		from textacy.preprocessing.replace import replace_urls, replace_emails, replace_numbers

		text = tok
		text = replace_urls(text, replace_with=replace_with)
		text = replace_emails(text, replace_with=replace_with)
		text = replace_numbers(text, replace_with=replace_with)
		return text
