#!/usr/bin/env python3

import csv

import random

from collections import namedtuple, Counter

from typing import List, Dict, NamedTuple

EMMSourceItem = namedtuple("EMMSourceItem", ['state', 'title', 'url', 'description', 'region', 'country_code', 'country', 'language_code', 'language', 'category', 'encoding', 'format', 'ranking', 'subject', 'type', 'period', 'frequency', 'tier'])

class EMMSourcesDB:

	"""
	manages a list of EMM sources

	main goal of the class is to conveniently filter and sample them
	"""

	def __init__(self, sources_fp):
		""" initialise source DB with csv file given as parameter """
		self.list_sources : List[EMMSourceItem] = []

		with open(sources_fp) as csvfile:
			reader = csv.reader(csvfile, delimiter=',')
			for row in reader:
				if len(row) == 0:
					continue
				if len(row) == 17:
					row.append(0)
				if row[0] == "state":
					continue
				sourceitem = EMMSourceItem(*row)
				
				self.list_sources.append(sourceitem)

	def getSources(self, filter_state="active", filter_name=None, filter_lang=None, filter_country=None, filter_region=None, filter_type=None, filter_encoding=None, filter_ranking=None, filter_tier=None, max_sources_per_country=None, max_sources_per_language=None, max_sources_per_country_language=None) -> List[EMMSourceItem]:
		""" extract list of sources based on different filters """

		# applies filters based on the sources' fields

		res = []

		for source in self.list_sources:
			if filter_state:
				if source.state != filter_state:
					continue

			if filter_name:
				if type(filter_name) != str:
					if source.title not in filter_name:
						continue
				elif source.title != filter_name:
					continue
			
			if filter_lang:
				if filter_lang == "Spanish":
					filter_lang = "Spanish, Castilian"
				
				if type(filter_lang) != str:
					if source.language_code not in filter_lang and source.language not in filter_lang:
						continue
				elif source.language_code != filter_lang and source.language != filter_lang:
					continue
			
			if filter_country:
				if type(filter_country) != str:
					if source.country not in filter_country and source.country_code not in filter_country:
						continue
				elif source.country != filter_country and source.country_code != filter_country:
					continue
					
			if filter_region:
				if source.region != filter_region:
					continue
			
			if filter_type:
				if type(filter_type) != str:
					if source.type not in filter_type:
						continue
				elif source.type != filter_type:
					continue

			if filter_encoding:
				if source.encoding != filter_encoding:
					continue

			if filter_ranking:
				if type(filter_ranking) != str:
					if source.ranking not in filter_ranking:
						continue
				elif source.ranking != filter_ranking:
					continue

			if filter_tier:
				if type(filter_tier) != str:
					if source.tier not in filter_tier:
						continue
				elif source.tier != filter_tier:
					continue

			res.append(source)

		# apply restrictions based on max number of sources

		if max_sources_per_country or max_sources_per_language or max_sources_per_country_language:
			map_lang_sources: Dict[str, List] = {}
			map_country_sources: Dict[str, List]  = {}
			map_countrylang_sources: Dict[str, List]  = {}

			for source in res:
				lang = source.language_code
				country = source.country_code

				if lang not in map_lang_sources:
					map_lang_sources[lang] = []
				map_lang_sources[lang].append(source)

				if country not in map_country_sources:
					map_country_sources[country] = []
				map_country_sources[country].append(source)

				key = country+"_"+lang
				if key not in map_countrylang_sources:
					map_countrylang_sources[key] = []
				map_countrylang_sources[key].append(source)

			def sort_ranking(list_src):
				return sorted(list_src, key=lambda x: x.ranking, reverse=True)

			res = []

			if max_sources_per_country:
			
				for country in map_country_sources:
					res_tmp = map_country_sources[country]
					random.shuffle(res_tmp)
					res_tmp = sort_ranking(res_tmp)
					res.extend(res_tmp[:max_sources_per_country])

			elif max_sources_per_language:
			
				for lang in map_lang_sources:
					res_tmp = map_lang_sources[lang]
					random.shuffle(res_tmp)
					res_tmp = sort_ranking(res_tmp)
					res.extend(res_tmp[:max_sources_per_language])

			elif max_sources_per_country_language:
			
				for country_lang in map_countrylang_sources:
					res_tmp = map_countrylang_sources[country_lang]
					random.shuffle(res_tmp)
					res_tmp = sort_ranking(res_tmp)
					res.extend(res_tmp[:max_sources_per_country_language])

		return res

	def exportToCSV(self, fp, list_sources=None):
		""" exports DB to csv or the list in parameters if provided """

		if list_sources is None:
			list_sources = self.list_sources
		
		with open(fp, "w") as f:
			csv_writer = csv.writer(f)
			csv_writer.writerow(['state', 'title', 'url', 'description', 'region', 'country_code', 'country', 'language_code', 'language', 'category', 'encoding', 'format', 'ranking', 'subject', 'type', 'period', 'frequency'])
			csv_writer.writerows([x for x in list_sources])

	def getStatistics(self):

		field_list = ['state', 'title', 'url', 'description', 'region', 'country_code', 'country', 'language_code', 'language', 'category', 'encoding', 'format', 'ranking', 'subject', 'type', 'period', 'frequency']

		for field_name in field_list:
			counter = Counter([getattr(source, field_name) for source in self.list_sources])

			print("==", field_name)
			print(counter.most_common())

		map_country_lang = {}
		map_lang_country = {}

		for source in self.list_sources:
			country = source.country_code
			lang = source.language_code

			if country not in map_country_lang:
				map_country_lang[country] = Counter()
			map_country_lang[country].update([lang])

			if lang not in map_lang_country:
				map_lang_country[lang] = Counter()
			map_lang_country[lang].update([country])

		print("==== country to lang")
		for key, counter in sorted(map_country_lang.items()):
			print(key, len(counter), counter.most_common())

		print("==== lang to country")
		for key, counter in sorted(map_lang_country.items()):
			print(key, len(counter), counter.most_common())
