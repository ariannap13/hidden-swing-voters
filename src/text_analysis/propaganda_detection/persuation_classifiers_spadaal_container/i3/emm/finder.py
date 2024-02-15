import json
from lxml import etree
import requests
import xmltodict
import urllib.parse
import time
import sys
import re
import tqdm
import math
import argparse
import os
from typing import List
import random

from datetime import datetime, timedelta
import dateutil.parser

import pandas as pd
import itertools

from i3.emm.objects import EMNNewsItem
from i3.emm.parser import EMMParser

class OutputManager:

	"""
	This class is used a programable callback for handling the displaying/saving of extracted data

	It is versatile, so that it either:
	- use a filename
	- use a file descriptor
	- use a file pattern, incrementing the counter with each call
	- use no file at all, but accumulate data in an internal list instead

	It also gives the possibility to:
	- preprocess data with a another callback
	- convert data to json before exporting
	"""

	def __init__(self, fd=None, fname=None, fname_pattern=None, accumulator=None, append=False, verbose=False, data_transform_function=lambda x: x, json_output=False):
		"""
		Parameters
		fd: file object where to write
		fname: fname to open and write, one file for all the data
		fname_pattern: fname to open and, one file per call to process
		accumulator: no IO, data is stored in a list
		verbose: verbose
		data_transformation_function: function to prepocess the data before processing
		json_output: whether to dump data to json before processing

		order preference: accumulator, fd, fname, fname_pattern
		"""
		self.fd = fd
		self.fname = fname
		self.fname_pattern = fname_pattern
		if self.fname_pattern:
			extension = ".xml" if not json_output else ".json"
			self.fname_pattern = self.fname_pattern if extension in fname_pattern else fname_pattern+extension
		self.counter = 0
		self.accumulator = accumulator
		self.verbose = verbose
		self.data_transform_function = data_transform_function
		self.json_output = json_output

		if self.fd:
			pass
		elif self.fname is not None and self.fname:
			self.fd = open(self.fname, "w" if not append else "a", encoding="utf-8")
		else:
			pass

	def process(self, x):
		""" process the data """
		if self.verbose:
			print("OutputManager: count", self.counter)

		# pre-processing
		data = self.data_transform_function(x)
		data = data if not self.json_output else json.dumps(data)

		# store to list
		if self.accumulator is not None:
			self.accumulator.extend(data)
		# write to file
		else:
			# write to file descriptor
			if self.fd:
				print(data, file=self.fd)
				self.fd.flush()
			# write to a new file for each call using a pattern and the counter
			else:
				fname = self.fname_pattern % self.counter 
				with open(fname, "w", encoding="utf-8") as f:
					print(data, file=f)
		self.counter += 1


class Finder:

	"""
	Query finder and either:
	- save data as raw xml files
	- save data as json files
	- parse data as a list of EMMNewsItem objects

	Default behaiour is to return list of objects, data is saved only when specifying an output manager
	It can be used to extract any amount of data when using in cunjunction with OutputManager
	"""

	# the main task behind querying the Finder is to form the url to query it
	# the url is made of four different part
	# - the server (which finder to use)
	# - the query 
	# - the filter (use for caching)
	# - the option (handle sorting and pagination)

	# query:
	# - start wieth "&q=""
	# - elements are separated by space
	# - the characters "()[]&=" must not be urlencoded, meaning that it is not possible to directly urlencode a query, but it must be construcred step by step
	# - having a "+" before a parameter name (like "country" or "language") means that it is mandatory, and acts as an "and"
	# - keyword search uses this construct: {!babel t="list of keywords"}, spaces inside keywords must be replaced by "+""
	# - all list of elements are inside parenthese, elements are space spearated. Default behavioir is OR search, forcing in element in with "+" prefixe and forcing out with "-" prefix

	# exemples of correct query-url pairs:
	# +country:MT {!babel t="Macron"} pubdate:[2020-01-14T00:00:00Z TO 2020-01-24T00:00:00Z]
	# http://139.191.34.85/Finder/Finder?op=search&q=%2Bcountry%3AMT%20%7B!babel%20t%3D%22Macron%22%7D%20pubdate%3A%5B2020-01-14T00%3A00%3A00Z%20TO%202020-01-24T00%3A00%3A00Z%5D

	# to donwnload more than 10000 pages, pagination is mandatory:
	# needs to both 1) set the cursorMark=* argument and 2) specifie a sorting order

	# predefined parameters lists
	list_eu_countries = "AT BE BG CY CZ DE DK EE ES FI FR GR HR HU IE IT LT LU LV MT NL PL PT RO SE SI SK".split()
	list_g8_countries = "CA FR DE IT JP US GB RU".split()

	def __init__(self, server=85, tunnel=False, project_name="unspecified"):
		""" initialise connection parameters

		server: either: full ip, last ip number, url, name ("emm", "disinfo", etc.)
		tunnel: use http tunnel or not
		project_code: mandatory field in order to obtain fulltext
		"""

		# process server option

		if type(server) == str:
			# full ip address
			if "." in server:
				server = server
			# description
			elif server.isalpha:
				if server == "emm":
					server = "139.191.34.85"
				elif server == "disinfo":
					server = "139.191.34.117"
				elif server == "medisys":
					server = "139.191.33.144"
				elif server == "emm4u":
					server = "https://data.emm4u.eu/"
				else:
					raise Exception("unrecognized server name: "+server)
		# last digits of the ip address
		else:
			server = "139.191.34."+str(server)

		server = "http://"+server if "http" not in server else server

		self.url_prefix = ("https://t1.emm4u.eu/EMMApp/Tunnel?url=" if tunnel else "") + \
							server + ("/Finder/Finder" if "/Finder" not in server else "")

		#self.query_postfix_elements = ["sort=pubdate%20asc,guid%20asc", "fulltext="+project_name]
		#self.query_postfix_elements = ["sort=guid%20asc", "fulltext="+project_name]
		self.query_postfix_elements = ["sort=pubdate%20asc,guid%20asc", "fulltext="+project_name]


	@staticmethod
	def urlEncode(x):
		return urllib.parse.quote(x).replace("[", "%5B").replace("]", "%5D")

	@staticmethod
	def generateDates(start_date, end_date, frequency="daily"):
		""" generate list of milestones days

		use case: when we can to download a given amount of data between milestones, whatever the exact day
		
		start_date: start date
		end_date: end_date
		frequency: one of: yearly, quarterly, monthly, daily """

		freq = None
		if frequency == "yearly":
			freq = "Y"
		elif frequency == "quarterly":
			freq = "Q"
		elif frequency == "monthly":
			freq = "M"
		elif frequency == "weekly":
			freq = "W"
		elif frequency == "daily":
			freq = "D"
		else:
			raise Exception("Invalid parameter, frequency:", frequency)

		list_milestones = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq=freq).strftime("%Y-%m-%d").tolist()

		return list_milestones

	@staticmethod
	def generateDateBoundaries(start_date, end_date, frequency="daily"):

		list_dates = Finder.generateDates(start_date, end_date, frequency)

		list_boundaries = []

		for idx_date in range(len(list_dates)-1):
			milestone_start = list_dates[idx_date].replace("/","-")
			milestone_end = list_dates[idx_date+1].replace("/","-")

			list_boundaries.append([milestone_start, milestone_end])

		return list_boundaries
		

	def buildQueryListByDate(self, query=None, keyword=None, source=None, language=None, country=None, date=None, date_start=None, date_end=None, frequency=None, category=None, guid=None, randomized_guid_for_maxitems=None):
		""" build a list of queries, caracterized by specific date interval """

		list_queries = []

		if frequency is None or frequency == "exact_date":
			query = self.buildQuery(query, keyword, source, language, country, date, date_start, date_end, frequency, category, guid, randomized_guid_for_maxitems)
			list_queries.append(query)
		else:
			list_dates = Finder.generateDates(date_start, date_end, frequency)

			print("DATES", list_dates)

			for idx_date in range(len(list_dates)-1):
				milestone_start = list_dates[idx_date].replace("/","-")
				milestone_end = list_dates[idx_date+1].replace("/","-")

				list_queries.append(self.buildQuery(query, keyword, source, language, country, date, milestone_start, milestone_end, frequency, category, guid, randomized_guid_for_maxitems))

		return list_queries


# TODO (Leo)
#said this, i think if possible the best would be that 
#-the user tells you startdate= enddate= [done]
#-you do a loop day by day putting the day like this [done]
#fq=pubdate:[2020-01-03T00:00:00Z TO 2020-01-03T23:59:59Z] [done]
#-by consequence the sort is just sort=guid asc [todo]

	def buildQuery(self, query=None, keyword=None, source=None, language=None, country=None, date=None, date_start=None, date_end=None, frequency=None, category=None, guid=None, randomized_guid_for_maxitems=None, force_post=False):
		""" build a query by specifing each filters. Each filter can be either a python list or a string (Finder syntax can be used freely)

		e.g.

		language = ["fr", "it"]    will result in   "language%3D(fr%20it)"
		language = "(fr it)"       will restult in   "language%3D(fr%20it)"

		Parameters
		query: exact query
		...

		Returns
		url
		"""

		# This code is complicated, because it supports different style of querying:
		# - by constructing a query which is either an url for GET or parameter list for POST
		# - by supporting two different solr query styles, the text-based filter querying (with +
		#     and - in front of operators) and the surround style of querying, where all parameter
		#     for filtering query must be encoded as sucessive fq and no spaces allowed betwen an operator and the value
		# also, the function is polymorphic: its argumentcan be either string or list

		# TODO:
		# - ajouter la possibiltie de faire des fq positive et negative par rapport a du text qui obligatoirement
		#     etre absent ou prsent dans le text
		#		&fq=+%7B!babel%20t%3D"vingtaine"%7D
		#		&fq=-%7B!babel%20t%3D"vingtaine"%7D

		use_text_fq = True #not force_post and query is None and "surround" not in query
		operator_prefix_get = "%2B" if use_text_fq else "&fq="
		operator_prefix = "\n+" if use_text_fq else "&fq="

		use_post = query is not None and "{!surround" in query
		use_post = False

		query_params = {}
		query_string = ""

		query_params["op"] = "search"

		def join_param(x):
			""" this function allows parameters to be processed either as python list or a string """ 
			return "(" + " ".join(x) + ")" if type(x) == list else "("+str(x)+")"

		query_elements = [] # contains elements to be added to query with &q
		filter_elements = [] # contains elements to be added to query with &fq

		has_text_query=False

		# directly fully specify query
		if query is not None:

		#
		# build query from arguments
		#
			query_elements += [urllib.parse.quote_plus(query)]

			#query_params["q"] = query

			query_string = query

		# process keywords

		if keyword is not None:
			# the url encoding is done differently than the other parameters
			if type(keyword) == list:
				keyword = [ x.replace(" ", "+") for x in keyword]
			if type(keyword) == str:
				keyword = [keyword]
			keyword = [Finder.urlEncode(x) for x in keyword]
			keyword = " ".join(keyword)
			query_elements += ["%2B%7B!babel%20t%3D%22{}%22%7D".format(keyword)]

			query_string += operator_prefix+"{!babel t=\"(" + keyword +")\"} "

			has_text_query=True

		# process source

		if source is not None:
			filter_elements += [operator_prefix_get+"source%3a"+join_param(source)]

			query_string += operator_prefix+"source:"+join_param(source)


		# process languages

		if language is not None:
			filter_elements += [operator_prefix_get+"language%3a"+join_param(language)]

			query_string += operator_prefix+"language:"+join_param(language)


		# process countries

		if country is not None:
			if country == "EU":
				country = Finder.list_eu_countries
			if country == "G8":
				country = Finder.list_g8_countries
			filter_elements += [operator_prefix_get+"country%3a"+join_param(country)]

			query_string += operator_prefix+"country:"+join_param(country)


		# process categories

		if category is not None:
			#if type(category) == str:
			#	category = [category]
			filter_elements += [operator_prefix_get+"category%3a"+join_param(category)]

			query_string += "\n"+operator_prefix+"category:"+join_param(category)


		# process guid

		if guid is not None:

			if randomized_guid_for_maxitems:
				raise Exception("Incompatible arguments: guid, randomized_guid_for_maxitems")

			if type(guid) == str:
				filter_elements += [operator_prefix_get+"guid%3A%28"+guid+"%29"]
			if type(guid) == list:
				filter_elements += [operator_prefix_get+"guid%3A%28"+"%20".join(guid)+"%29"]

			query_params["guid"] = guid

		#
		# parse date
		#

		query_date = None

		# directly specified the date in solr format
		if date:
			if date == "*":
				# any date
				query_date = "pubdate%3a*"

				query_params["fq"] = "pubdate:*"

			elif "-" in date:
				# generate the whole day as a time range
				date_start = date+"T00:00:00Z"
				date_end = date+"T23:59:59Z"
				query_date = Finder.urlEncode("pubdate:["+date_start+" TO "+date_end+"]")

				query_params["fq"] = "pubdate:["+date_start+" TO "+date_end+"]"

			else:
				# use whatever is specified by the user
				query_date = "pubdate%3a"+date

				query_params["fq"] = "pubdate:"+date

		# specify start and end range
		elif date_end is not None or date_end is not None:

			#if frequency is None:
			#	raise Exception("using start_date and end_date require to define a frequency")

			if "T" in date_start and date_start == date_end:
				raise Exception("error in date parameter: "+ str(date_start)+" "+str(date_end))

			if "T" not in date_start and date_start != "*":
				date_start += "T00:00:00Z"

			date_start = dateutil.parser.parse(date_start).strftime(
				"%Y-%m-%dT%H:%m:%SZ") if (date_start is not None and date_start) != "*" else "*"

			if "T" not in date_end and date_end != "*":
				date_end += "T23:59:59Z"

			date_end = dateutil.parser.parse(date_end).strftime(
				"%Y-%m-%dT%H:%m:%SZ") if (date_end is not None and date_end != "*") else "*"

			query_date = "pubdate:["+date_start+" TO "+date_end+"]"

			query_params["fq"] = query_date

			query_date = Finder.urlEncode(query_date)

		if query_date is not None:
			filter_elements += [operator_prefix_get + query_date]


		#
		# build url
		#

		query_prefix = self.url_prefix + "?op=search"

		if len(query_elements) > 0:
			encoded_query = "&q=" + "%20".join(query_elements)
		else:
			encoded_query = "&q=*:*"

		#encoded_query = ""
		#if encoded_query:
		#	if len(encoded_query) > 1:
		#		encoded_query = "&q=" + "&fq=".join(filter_elements)
		#	else:
		#		encoded_query = "&q="+filter_elements[0]
			

		encoded_filter = ""
		if filter_elements:
			if len(filter_elements) > 1:
				encoded_filter = "&fq=" + "&fq=".join(filter_elements)
			else:
				encoded_filter = "&fq="+filter_elements[0]
			
		encoded_options = ""
		if self.query_postfix_elements:
			if len(self.query_postfix_elements) > 1:
				encoded_options = "&" + "&".join(self.query_postfix_elements)
			else:
				encoded_options = "&"+self.query_postfix_elements[0]

		query_url = query_prefix + encoded_query + encoded_filter + encoded_options

		if randomized_guid_for_maxitems:

			if not use_post:
#				print(query_url)
				guid_list = self.get_randomized_guidlist(query_url, randomized_guid_for_maxitems)
			else:
#				print(query_params)
				guid_list = self.get_randomized_guidlist(query_params, randomized_guid_for_maxitems)


			# for some reasons, needed to write "+&fq=" here instead of "+fq=" for this filter work
			guid_filter = "&fq=guid%3A("+"%20".join(guid_list)+")"

			filter_elements += [guid_filter]

#			query_params["guid"] = "\n"+operator_prefix+"guid: "+ join_param(guid_list)
			query_string = "\n"+operator_prefix+"+fq=guid:"+ join_param(guid_list)


			# recreate the query accounting for the news guid parameters

			#encoded_query = "&q=" + "%20".join(query_elements)

			query_url = query_prefix + encoded_query + encoded_filter + guid_filter+ encoded_options

		print(query_url)

		print("--")

		#if query is None:
		query_params["q"] = query_string

		print(query_params)

		if use_post:
			print("post")
			return query_params

		print("get")
		return query_url


	def get_randomized_guidlist(self, query, randomized_guid_for_maxitems, rand_seed=0):
		""" This is Guilaume's randomization algorithm, I can't understand it, but it works.
		if a maximum number of replies is specified, it will evenly randomize the sampling over all
		possibles items, by playing on the guid filter.
		
		modification randomization of the numbers to pick from, instead of only adding 1
		"""

		list_numbers = [0,1,2,3,4,5,6,7,8,9, "a", "b", "c", "d", "e", "f"]

		if rand_seed != 0:
			randomgen = random.Random(rand_seed).shuffle(list_numbers)
		else:
			randomgen = random
		
		randomgen.shuffle(list_numbers)

		if type(randomized_guid_for_maxitems) != int or int(randomized_guid_for_maxitems) > 10000:
			raise Exception("invalid value for randomized_guid_for_maxitems:", randomized_guid_for_maxitems)


		use_post = query is not None and "{!surround" in query
		use_post = False

		if not use_post:
#			print("get")
			r = requests.get(query+"&rows=0").text
		else:
#			print("post")
			query["rows"] = "0"
			r = requests.post(self.url_prefix, params=query).text

		nb_hits = self.getHitCount(r)

		print(nb_hits, randomized_guid_for_maxitems)

		ratio = nb_hits / randomized_guid_for_maxitems

		guid_list = ['*-*']
		if ratio > 2:
			guid_list = []
			iter_filter = ''
			ratio = ratio / 16
			while ratio > 1:
				randomgen.shuffle(list_numbers)
				number = list_numbers[0]
				iter_filter += str(number) # '1'
				ratio = ratio / 16
			nb_num = round(1 / ratio)
			if nb_num > 3:  # this condition shouldn't be used to be clean. Used because of some constraints in the size
							# of the finder query, in particular for the value of the guid attribute
							# Probably, it has been added because of my random queries ;)
				iter_filter = iter_filter[1:]
				nb_num = 1

			randomgen.shuffle(list_numbers)
			for i in range(nb_num):
				number = list_numbers[i]
				guid_list.append('*-{}{}*'.format(iter_filter, str(number)))
#				guid_list.append('*-{}{}*'.format(iter_filter, hex(numhber)[2:]))

		return guid_list

	@staticmethod
	def getHitCount(page):
		tot_hits = re.search(r"<numFound>([0-9]+)</numFound>", page)
		if tot_hits is not None:
			tot_hits = int(tot_hits.group(1))
		return tot_hits

	@staticmethod
	def getRows(page):
		rows = re.search(r"<rows>([0-9]+)</rows>", page)
		if rows is not None:
			rows = int(rows.group(1))
		return rows

	def fetchQuery(self, query, recursive=False, start=None, page_size=10, current_mark="*", output=None, quiet=False, hitcount_warning_threshold=10000, hitcount_cutoff_threshold=10000, raw=False):
		""" fetches data related to a query, if not output manager is specified, list of EMM objects is returned
		
		Parameters
		query: exact query 
		recursive: if True, downloads all the matching data by batch, if False only page_size
		start: from which rows to start fetching data
		page_size: number of rows to get with each call
		current_mark: solr token for pagination
		output: where to store the output (OutputManager object), if None returns list of parsed EMM objects
		quiet: quiet
		hitcount_warning_threshold: when downloading all data, warn the user if nb of rows higher than threshold
		hitcount_cutoff_threshold: cutoff the number of rows to download
		raw: return xml as as string instead of parsed objects

		Returns
		list_items: list of news items
		nb_rows: how many rows retrieved from Finder in current reply
		"""

		if type(query) == dict:
			query_params = dict(query)
			query = "no_get_query"
		else:
			query_params = {}

		### check boundaries

		if page_size > hitcount_warning_threshold:
			print("WARNING: high page_size", page_size, file=sys.stderr)

		if page_size > hitcount_cutoff_threshold:
			print("WARNING: high page_size, caping at max", page_size, hitcount_cutoff_threshold, file=sys.stderr)
			page_size = hitcount_cutoff_threshold

		### add optional pagination elements to the url

		fetch_postfix_elements = []  # , "timeAllowed=6000"

		if start is not None:
			fetch_postfix_elements.append("start="+str(start))

			query_params["start"] = str(start)

		if recursive:
			fetch_postfix_elements.append("cursorMark=*")

			query_params["cursorMark"] = "*"


		final_query = query + ("&" if fetch_postfix_elements else "") + \
			"&".join([x for x in fetch_postfix_elements])

		lookup_query = final_query # lookup query fetches default number of rows
		final_query += "&rows="+str(page_size) # final query fetches specified number of rows

		query_params["rows"] = "0"

		### perform first query ot get page count

		if recursive:
			print("RECURSIVE")
			print(lookup_query)

			r = requests.get(lookup_query)
#			r = requests.post(self.url_prefix, params=list(query_params.items()))

			nb_hits = self.getHitCount(r.text)
			nb_rows = self.getRows(r.text)

			if not quiet:
				print("total hits:", nb_hits, "rows:", nb_rows)

			if nb_hits is None or nb_rows is None:
				print("WARNING: empty reply")
				return 0, 0

			tot_calls = math.ceil(nb_hits / page_size)

			if tot_calls >= hitcount_warning_threshold:
				if not quiet:
					print("WARNING: high hit count", nb_hits, file=sys.stderr)

			if tot_calls >= hitcount_warning_threshold:
				if not quiet:
					print("WARNING: hit count above max, capping at max",
						hitcount_cutoff_threshold, file=sys.stderr)
				tot_calls = hitcount_cutoff_threshold
		else:
			tot_calls = 1

		query_params["rows"] = str(page_size)

		### setup output

		#list_items: List[EMNNewsItem] = []
		list_items = []

		# if no output manager is defined, return parsed EMM object, unless raw is specified
		if output is None:
			if not raw:
				output = OutputManager(accumulator=list_items, data_transform_function=EMMParser.parseXmlPage)
			else:
				output = OutputManager(accumulator=list_items)
		### perform all the queries

		last_mark = None

		orig_url = str(final_query)
		next_url = orig_url

		progress_bar = tqdm.tqdm if not quiet and recursive else lambda x: x

		tot_rows = 0

		for count_rows in progress_bar(range(tot_calls)):

			print("== call nb", count_rows)

			print("URL / PARAMS", next_url, query_params)

			# fetch data

			if query != "no_get_query":
				print("GET")
				print(next_url)
				r = requests.get(next_url)
			else:
				print("POST")
				print(query_params)
				r = requests.post(self.url_prefix, params=list(query_params.items()))

			count_rows += 1

			print(r.status_code)
			print(r.text[:200])

			# perform output

			output.process(r.text)

			# process pagination

			if recursive:
				datadict = xmltodict.parse(r.text)
				last_mark = current_mark
				current_mark = datadict["rss"]["channel"]["nextCursorMark"] if "nextCursorMark" in datadict["rss"]["channel"] else None
				current_mark = urllib.parse.quote(current_mark) if current_mark else None

				print("mark:", current_mark)

				if not (last_mark != current_mark and current_mark != None):
					if not quiet:
						print("WARNING: no next mark, exiting at iteration", count_rows)
					break

				next_url = orig_url.replace("*", current_mark)

				query_params["cursorMark"] = current_mark

			# stats

			nb_rows = self.getRows(r.text)
			tot_rows += nb_rows if nb_rows is not None else 0

			# just in case
			time.sleep(1)

		### display info on last fetch

		nb_hits = self.getHitCount(r.text)
		nb_rows = self.getRows(r.text)

		print("RAW", raw)

		if not quiet:
			print("total hits:", nb_hits, "total rows", tot_rows, "last rows:", nb_rows)

		if nb_hits is None or nb_rows is None:
			print("WARNING: empty reply")
			return [], 0

		if raw:
			list_items = "".join(list_items)


		return list_items, nb_hits


class BatchDownloader:

	"""
	Generated several queryies given one basic query, saving the results under different files.
	results are goruped by a set of fields

	e.g. grouping by countries and category means that each (country, category) pair will be fetched and saved separately
	with distinctive filename

	Groups by the list of parameters defined in the parameters! Not the value of these fields in the answers
	"""


	all_fields = set(["language", "keyword", "country", "source", "category", "date", "date_start", "date_end", "query"])

	def __init__(self, finder):

		self.finder = finder

	@staticmethod
	def generateDateRange(str_start, str_end, generate_string=False):
		""" generate a range of date, with 1 day step """

		list_dates = pd.date_range(start=str_start,end=str_end).to_pydatetime().tolist() # type: ignore

		if generate_string:
			list_dates = [x.strftime("%Y-%m-%d") for x in list_dates]

		return list_dates

	def fetchAndSaveGroups(self, params, max_fields, prefix=None, recursive=False, page_size=None, use_json=False):
		""" generate queries for all combinations of group-by parameters and save them in separate files following predefined naming pattern """

		combinations = itertools.product(*[params[name] for name in max_fields])

		sum_fields = set(BatchDownloader.all_fields) - max_fields

		for comb in combinations:
			comb_params = dict(params)

			print("====")
			print(comb)
			
			for field_name, field_value in zip(max_fields, comb):
				#print(field_name, field_value)
				comb_params[field_name] = field_value

			print(comb_params)

			filename = [ comb_params[x] if x in max_fields else ("all" if comb_params[x] is not None else "any") for x in BatchDownloader.all_fields]
			filename = "_".join(filename)
			filename = (prefix if prefix is not None else "") + filename + (".xml" if not use_json else ".json")

			print(filename)
			om = OutputManager(fname=filename)

			for field_name in BatchDownloader.all_fields:
				if comb_params[field_name] is None:
					del comb_params[field_name]

			print(comb_params)

			url = self.finder.buildQuery(**comb_params)

			print(url)

			self.finder.fetchQuery(url, recursive=recursive, output=om, page_size=page_size)

