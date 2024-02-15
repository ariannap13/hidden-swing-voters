

import json
import xmltodict
import dateutil

import sys 

from typing import List

from collections import OrderedDict

from i3.emm.objects import *
from i3.nlp.corpus import TextDoc, FileAdapter



class EMMParser:
	"""
	Class to parse raw xml finder dumps to python objects
	all methods are static, use EMMAdapter to instanciate
	"""

	def __init__(self):
		pass

	@staticmethod
	def parseXmlFile(filename: str, default_sentiment_classifier="svm-v1"):
		""" parse a file an xml dump file """

		# the file can contain several xml file concatenated to each others
	
		list_items: List[EMNNewsItem] = []

		with open(filename, encoding='utf-8') as f:
			list_items = EMMParser.parseXmlPage(f.read(), default_sentiment_classifier)

		return list_items

	@staticmethod
	def parseXmlPage(page: str, split=True, default_sentiment_classifier="svm-v1"):
		""" parse a string containing emm xml to emm python objects """

		split_token = '<?xml version="1.0" encoding="UTF-8"?>'

		list_items: List[EMNNewsItem] = []

		if split == False:
			datadict = xmltodict.parse(page)

			if "rss" not in datadict or "channel" not in datadict["rss"] or "item" not in datadict["rss"]["channel"]:
				return []

			if type(datadict["rss"]["channel"]["item"]) != list:
				datadict["rss"]["channel"]["item"] = [datadict["rss"]["channel"]["item"]]

			for item in datadict["rss"]["channel"]["item"]:
				emmitem = EMMParser.parseItem(item, default_sentiment_classifier)
				if emmitem is not None:
					list_items.append(emmitem)

			return list_items
		else:
			rss_list = page.split(split_token)
			for rss in rss_list:
				if len(rss) == 0:
					continue
				list_items.extend(EMMParser.parseXmlPage(split_token+rss, split=False, default_sentiment_classifier=default_sentiment_classifier))
			return list_items


	@staticmethod
	def parseItem(item, default_sentiment_classifier="svm-v1"):
		""" parse xml item """
		if not isinstance(item, OrderedDict):
			return None

		if "iso:language" not in item:
			return None

		lang = item["iso:language"]
		title = item["title"]
		if title is None: title = ""
		description = item["description"]
		if description is None: description  = ""
		text = item["emm:text"]["#text"].replace("\t", "   ").replace("\n", "   ") if "emm:text" in item and "#text" in item["emm:text"] else ""
		date_str = item["pubDate"]
		#date_parsed = dateutil.parser.parse(date_str)
		link = item["link"] if "link" in item else None
		guid = item["guid"]
		content_type = item["emm:contentType"] if "emm:contentType" in item else None

		duplicate = item["@duplicate"] if "@duplicate" in item else None


		def get_if_exists(k, m):
			if m is not None:
				return m[k] if k in m else None
			return None

		tag = get_if_exists("emm:title", item)
		title_en = get_if_exists("#text", tag) if get_if_exists("@lang", tag) == "en" else "" 

		tag = get_if_exists("emm:description", item)
		description_en = get_if_exists("#text", tag) if get_if_exists("@lang", tag) == "en" else ""  

		tag = get_if_exists("emm:translate", item)
		text_en = get_if_exists("#text", tag) if get_if_exists("@lang", tag) == "en" else ""  

		## category

		try:
			category = [parseEMMCategory(x) for x in item["category"]] if "category" in item else None
		except:
			category = [parseEMMCategory(item["category"])]

		## geo tag

		list_geo_elems = []
	#	list_geo_elems.extend([x for x in item["emm:georss"]] if "emm:georss" in item else [])
	#	list_geo_elems.extend([x for x in item["emm:fullgeo"]] if "emm:fullgeo" in item else [])

		georss = item["emm:georss"] if "emm:georss" in item else None
		if georss is None:
			georss = []
		elif type(georss) == list:
			pass
		else:
			georss = [georss]

		georss_obj = [parseEMMGeo(x) for x in georss]

		fullgeo = item["emm:fullgeo"] if "emm:fullgeo" in item else None
		if fullgeo is None:
			fullgeo = []
		elif type(fullgeo) == list:
			pass
		else:
			fullgeo = [fullgeo]

		fullgeo_obj = [parseEMMGeo(x) for x in fullgeo]

		## source

		try:
			source = EMMSource(*[x for x in item["source"].values()])
		except:
			source = parseEMMSource(item["source"])

		## entity

		try:
#			entity = [EMMEntity(*ent.values()) for ent in item["emm:entity"]] if "emm:entity" in item else None
			entity = [parseEMMEntity(ent) for ent in item["emm:entity"]] if "emm:entity" in item else None
		except:
#			entity = [pEMMEntity(item["emm:entity"].values())]
			ent_obj = item["emm:entity"]
			if type(ent_obj) == list:
				entity = [parseEMMEntity(ent) for ent in ent_obj]
			else:
				entity = [parseEMMEntity(ent_obj)]

		if entity is not None:
			# in simplified finders, the count is not reported, hence the complex sorting function
			entity = sorted(entity, key=lambda x: x.count if hasattr(x, "count") and x.count is not None else 0, reverse=True)
		else:
			entity = []

		## sentiment

		# ('emm:sentiment', [OrderedDict([('@mode', 'svm-v1'), ('#text', 'neutral')]), OrderedDict([('@mode', 'xlm-v1'), ('#text', 'negative')])])

		sentiment = item["emm:sentiment"] if "emm:sentiment" in item else None
		if sentiment is not None:
			if type(sentiment) == list:
				# iterate to find the one corresponding to "svm-v1", which is current prefered one
				# if not present, return whatever is the other classifier present
				for sent in sentiment:
					if sent["@mode"] == default_sentiment_classifier:
						sentiment = sent
						break
			else:
				sent = sentiment
			sentiment = sent["#text"]

		emotion = item["emm:emotion"]["#text"] if "emm:emotion" in item else None
		tonality = int(item["emm:tonality"]) if "emm:tonality" in item else None

		## quotes
		if 'emm:quote' in item:
			qt_obj = item["emm:quote"]
			if type(qt_obj) == list:
				quotes = [parseEMMQuote(qt) for qt in qt_obj]
			else:
				quotes = [parseEMMQuote(qt_obj)]
		else:
			quotes = None

		## links

		def parseEMMLink(x):
			return [x["@href"], x["#text"]]

		if 'emm:link' in item:
			qt_obj = item["emm:link"]
			if type(qt_obj) == list:
				links = [parseEMMLink(qt) for qt in qt_obj]
			else:
				links = [parseEMMLink(qt_obj)]
		else:
			links = []


		emni = EMNNewsItem(lang, title, description, text, date_str, source, entity, sentiment, emotion, tonality, category, georss_obj, fullgeo_obj, content_type, link, guid, title_en, description_en, text_en, duplicate, quotes, links, {})

		return emni

	@staticmethod
	def displayPageElements(fp):
		""" display all the content of a file, for visual inspection/debugging """

		with open(fp,encoding='utf-8' ) as f:

			datadict = xmltodict.parse(f.read())

			#current_mark = datadict["rss"]["channel"]["nextCursorMark"] if "nextCursorMark" in datadict["rss"]["channel"] else None

			print(datadict.keys())
			print(datadict["rss"].keys())
			for k in datadict["rss"].keys():
				d = datadict["rss"][k]
				print(k, "=>", d.keys() if type(d) == dict or isinstance(d, OrderedDict) else type(d))

			print("\n== CHANNEL ==\n")

			for k in datadict["rss"]["channel"].keys():
				d = datadict["rss"]["channel"][k]
				print(k, "=>", d.keys() if type(d) == dict or isinstance(d, OrderedDict) else type(d))

				for k in datadict["rss"]["channel"]["item"]:
					print(type(k), k.items() if type(k) == OrderedDict else "")

			print("\n== ITEMS ==\n")
			for item in datadict["rss"]["channel"]["item"]:

				print("\n")
				for key, d in item.items():
					if key != "emm:text":
						print(key, "=>", d)
					else:
						d = d["#text"]
						t = d[:50] + " ... " + d[-50:] if len(d) > 20 else d
						print(key, "=>", t)


class EMMAdapter(FileAdapter):
	"""	
	class to parse xml file dump from Finder and store them as a TextCorpus object

	It is mandatory in the constructor to specify which field of news item will
	be extracted as text and which fields will be extracted as tags
	"""

	def __init__(self, fp=None, page=None, extract_categories=False,
						extract_entities=False,
						extract_text=False,
						extract_description=False,
						extract_title=False,
						tag_emotions=False):
		self.emmparser = EMMParser()

		self.extract_categories = extract_categories
		self.extract_entities = extract_entities
		self.extract_text = extract_text
		self.extract_title = extract_title
		self.extract_description = extract_description
		self.tag_emotions = tag_emotions

		self.documents = []

		if fp is not None:
			self.readFile(fp)

		if page is not None:
			self.processPage(page)

	def readFile(self, fp):
		""" parse a file """

		print("reading: ", fp)

		with open(fp, encoding='utf-8') as f:
			list_items: List[EMNNewsItem] = self.emmparser.parseXmlPage(f.read())
	
		return self.processItems(list_items)

	def processPage(self, page):
		""" parse xml content """
		list_items = self.emmparser.parseXmlPage(page)
		self.processItems(list_items)

	def processItems(self, list_items):
		""" extract data from xml to python object """
		
		documents = self.documents
		documents = []
		
		for emmitem in list_items:
			#print(emmitem)
			doc = []
			tags = {}
			if self.extract_entities: 
				if emmitem.entity is not None:
					for emment in emmitem.entity:
						enttxt = emment.text
						enttxt = enttxt.replace(" ", "_")
						enttxt = enttxt.replace("\n", "")
						doc.append(enttxt)
			if self.extract_text:
				doc.append(emmitem.text)
			if self.extract_description:
				doc.append(emmitem.description)
			if self.extract_title:
				doc.append(emmitem.title)
			if self.tag_emotions:
				tags["sentiment"] = emmitem.sentiment
				tags["emotion"] = emmitem.emotion
				tags["tonality"] = emmitem.tonality

			#print(doc)
			if len(doc) > 0:
				doc = " ".join(doc)
				doc = TextDoc(doc, tags=tags)
				documents.append(doc)
		
		return documents


if __name__ == "__main__":
	import sys

	emmparser = EMMParser()

	list_item = emmparser.parseXmlFile(sys.argv[1])

	print(list_item)
	print(len(list_item))
