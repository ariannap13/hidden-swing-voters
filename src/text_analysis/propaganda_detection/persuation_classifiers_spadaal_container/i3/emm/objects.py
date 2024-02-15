
from collections import namedtuple

"""
This class contains the code to parse structured fields of EMM news items
- Entity
- Source
- Category
- Georsss (most precise guess of the location of the news)
- Fullgeo (all the different location refered to in the news)

n.b. Fullgeo and Entity are complementary

ps: first letter of geoentity id describes if it s a city or country, etc.
"""

EMNNewsItem = namedtuple("EMMNewsItem", ["lang", "title", "description", "text", "date", "source", "entity", "sentiment", "emotion", "tonality", "category", "georss", "fullgeo", "content_type", "link", "guid", "title_en", "description_en", "text_en", "duplicate", "quotes", "links", "meta"])
EMMSource = namedtuple("EMMSource", ["url", "country", "id"])
EMMEntity = namedtuple("EMMEntity", ["id", "type", "subtype", "count", "pos", "name", "text"])
EMMCategory = namedtuple("EMMCategory", ["name", "rank", "score", "trigger"])
EMMGeo = namedtuple("EMMGeo", ["name", "id", "iso", "lat", "long", "count", "pos", "charpos", "wordlen", "score", "text"])
EMMQuote = namedtuple("EMMQuote",["who","verb","text","categories"])

def parseItem(lang, title, description, text, date, source, entity, sentiment, emotion, tonality, category, georss, fullgeo, content_type, link, guid, title_en, description_en, text_en, duplicate, quotes, links, meta={}):
	""" transform a json item, or a list of list, into a nammed tuple """
	source = EMMSource(*source)
	entity = [EMMEntity(*x) for x in entity]
	category = [EMMCategory(*x) for x in category]
	georss = [EMMGeo(*x) for x in georss]
	fullgeo = [EMMGeo(*x) for x in fullgeo]
	quotes = [EMMQuote(*x) for x in quotes] #untested
	links = links

	item = EMNNewsItem(lang, title, description, text, date, source, entity, sentiment, emotion, tonality, category, georss, fullgeo, content_type, link, guid, title_en, description_en, text_en, duplicate, quotes, links, meta)

	return item

def parseEMMEntity(item):
	data_ent = { fn:None for fn in ["id", "type", "subtype", "count", "pos", "name", "text"]}
	# assumes that field name in XML file and EMMEntity object are the same
	for k, v in item.items():
		# remove non alpha character to have xml and object fields correspond to each other
		k = "".join([x for x in k if x.isalpha()])
		data_ent[k] = v
	# delete buggy, unsuported or redundant info
	for key in ["functions", "sentiment"]:
		if key in data_ent:
			del data_ent[key]
	# convert numeric fields to int
	if data_ent["count"] is not None:
		data_ent["count"] = int(data_ent["count"])
	if data_ent["pos"] is not None:
		data_ent["pos"] = [int(x) for x in data_ent["pos"].split(",")]
	# create entity
	entity = EMMEntity(*data_ent.values())
	return entity

def parseEMMSource(item):
	data_src = { fn:None for fn in ["url", "country", "id"]}
	# deal with key shared between object
	for key in ["url", "country"]:
		if key in item:
			data_src[key] = item[key]
	# deal with key with different names than the fields
	if "#text" in item:
		data_src["id"] = item["#text"]
	if "@country" in item:
		data_src["country"] = item["@country"]
	if "@url" in item:
		data_src["country"] = item["@url"]
	source = EMMSource(*data_src.values())
	return source

def parseEMMCategory(item):
	data_src = { fn:None for fn in ["name", "rank", "score", "trigger"]}
	if "#text" in item:
		data_src["name"] = item["#text"]
	if "@emm:rank" in item:
		data_src["rank"] = int(item["@emm:rank"])
	if "@emm:score" in item:
		data_src["score"] = int(item["@emm:score"])
	if "@emm:trigger" in item:
		data_src["trigger"] = [ x.strip() for x in item["@emm:trigger"].split(";") if len(x.strip()) > 0 ]
	category = EMMCategory(*data_src.values())
	return category

def parseEMMGeo(item):
	data_src = { fn:None for fn in ["name", "id", "iso", "lat", "lon", "count", "pos", "charpos", "wordlen", "score", "text"]}
	for key in data_src.keys():
		if "@"+key in item:
			data_src[key] = item["@"+key]
		elif "#"+key in item:
			data_src[key] = item["#"+key]
	geotag = EMMGeo(*data_src.values())
	return geotag

def parseEMMQuote(item):
	data_src = {fn:None for fn in ["who","verb","text","categories"]}

	# deal with key with different names than the fields
	if "#text" in item:
		data_src["text"] = item["#text"]
	if "@who" in item:
		data_src["who"] = item["@who"]
	if "@verb" in item:
		data_src["verb"] = item["@verb"]
	if "@categories" in item:
		data_src['categories'] =  item["@categories"]
	quote = EMMQuote(*data_src.values())
	return quote
