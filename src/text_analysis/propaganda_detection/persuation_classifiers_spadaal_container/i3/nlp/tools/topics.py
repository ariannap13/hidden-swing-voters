
import json
import sys

from i3.nlp.corpus import TextCorpus
from i3.nlp.textprocessor import TextProcessor
from i3.nlp.graphprocessor import GraphProcessor

from typing import List

import networkx as nx

#import community
from community import community_louvain

from cdlib import algorithms

"""
Objects
"""

class Topic:

	""" Reresent a topic as:
	- a list of keyword
	- a relative importance
	- a list of subtopics

	TODO:
	- add desciption in human language
	- add documents to the topic
	"""

	def __init__(self, keywords, importance, subtopics=[], measures={}):
		self.keywords = keywords
		self.importance = importance
		self.subtopics = subtopics
		self.measures = measures

	def toDict(self):
		""" transform Topic object into a dictionary """
		return { "keywords": self.keywords, "importance": self.importance, "subtopics": ([x.toDict() for x in self.subtopics] if len(self.subtopics) > 0 else []), "measures": self.measures}

	@staticmethod
	def fromDict(keywords, importance, subtopics, measures):
		""" create Topic object from dictionary """
		t = Topic(keywords, importance, [Topic.fromDict(x["keywords"], x["importance"], x["subtopics"], x["measures"]) for x in subtopics], measures)
		t.subtopics.sort(key=lambda x: x.importance, reverse=True)
		return t

	def __repr__(self):
		return str(self.toDict())

	def __str__(self):
		return str(self.toDict())

	@staticmethod
	def saveTopicList(list_topics, fp, output_dir="."):
		""" save list of topics to json """
		with open(output_dir+"/"+fp, "w") as f:
			json.dump([x.toDict() for x in list_topics], f)

	@staticmethod
	def loadTopicList(fp, input_dir="."):
		""" load list of topics from json """
		with open(input_dir+"/"+fp) as f:
			list = [Topic.fromDict(x["keywords"], x["importance"], x["subtopics"], x["measures"]) for x in json.load(f)]
			list.sort(key=lambda x: x.importance, reverse=True)
			return list

"""
Computation
"""


class LDATopicMiner:

	def __init__(self):
		pass

class HLDATopicMiner:

	def __init__(self):
		pass

class NMFTopicMiner:

	def __init__(self):
		pass


class TopicMiner:

	""" Regroup all the topic mining algorithms	"""

	def __init__(self):
		pass

	@staticmethod
	def pretty_node_list(nodes_items, prop=0., n=None):
		""" pretty print list of pairs (str, float) """
		final_nodes = [ (x,v) for x,v in nodes_items if v >= nodes_items[int(len(nodes_items) * prop)][1]]
		return [ (w, "%.2f" % v) for w,v in final_nodes ][:n if n is not None else len(final_nodes)]

	@staticmethod
	def pretty_node_list2(nodes_items, n=None):
		""" pretty print list of pairs (str, float) """
		return [ (w, "%.2f" % v) for w,v in nodes_items ][:n if n is not None else len(nodes_items)]

	@staticmethod
	def processCorpus(corpus, min_degree, min_weight, linking_dist, resolution):
		""" process a corpus

		corpus: TextCorpus object
		min_degree: min degree of a node to be included
		min_weight: min weight of an edge to be included
		linking_dist: how far before/after a word shoudl be linked with its neighbours
		resolution: control how many clusters to topics should be (default: 1., < 1. more cluster, > 1. less clusters)
		"""

		graph = GraphProcessor.processCorpus(corpus, min_degree, min_weight, linking_dist)

		return TopicMiner.processGraph(graph, resolution)

	@staticmethod
	def processGraph(graph, resolution=1.):
		""" process a netorkx x graph
		
		graph: networkx graph
		resolution: control the amount of clusters """
		return TopicMiner.mine_lib_community(graph, resolution)

	@staticmethod
	def mine_lib_community(graph, resolution, rec_level=1, maximum_nodes_summary=10):
		""" use the community librairy to mine for node clusters

		graph: networkx graph
		resolution: control the amount of clusters """
	
		partition = community_louvain.best_partition(graph, resolution=resolution, randomize=False)

		results = []

		for com_idx, com in enumerate(set(partition.values())):
#			print("\n===", com_idx)
			list_nodes = [node for node in partition.keys() if partition[node] == com]
			
			subgraph = graph.subgraph(list_nodes)

			most_degree = GraphProcessor.getNodesSortedByDegree(subgraph)
			most_central = GraphProcessor.getNodesSortedByCentrality(subgraph)


			map_degree_centrality = nx.degree_centrality(subgraph)
			list_degree_centrality = sorted(map_degree_centrality.items(), key=lambda x: x[1], reverse=True)

			try:
				map_eigenvector_centrality = nx.eigenvector_centrality(subgraph)
				list_eigenvector_centrality = sorted(map_eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)
			except:
				list_eigenvector_centrality = [ (n, 0) for n in subgraph.nodes]

			map_degree = { x:graph.degree(x) for x in subgraph.nodes}
			list_degree = sorted(map_degree.items(), key=lambda x: x[1], reverse=True)

			map_clustering = nx.degree_centrality(subgraph)
			list_clustering = sorted(map_clustering.items(), key=lambda x: x[1], reverse=True)

			map_pagerank = nx.pagerank(subgraph)
			list_page_rank = sorted(map_pagerank.items(), key=lambda x: x[1], reverse=True)


			measures = {}
			measures["page_rank"] = list_page_rank
			measures["degree_centrality"] = list_degree_centrality
			measures["eigenvector_centrality"] = list_eigenvector_centrality
			measures["list_degree"] = list_degree
			measures["list_clustering"] = list_clustering

			map_node_centrality = {w:v for w,v in most_central}

			summary = set([ x[0] for x in most_degree[:maximum_nodes_summary]]) | set([ x[0] for x in most_central[:maximum_nodes_summary]]) | set([x[0] for x in list_page_rank[:maximum_nodes_summary]])
#			summary = sorted(list(summary), key=lambda x: map_node_centrality[x])
			summary = sorted(list(summary), key=lambda x: map_pagerank[x])


#			print("***", com_idx, len(list_nodes))
#			print("++++++")
#			print("Degree", TopicMiner.pretty_node_list2(list_degree[:10]))
#			print("Degree Centrality", TopicMiner.pretty_node_list2(list_degree_centrality[:10]))
#			print("Eigenvector Centrality", TopicMiner.pretty_node_list2(list_eigenvector_centrality[:10]))
#			print("Clustering", TopicMiner.pretty_node_list2(list_clustering[:10]))
#			print("------")
#			print("Degree", TopicMiner.pretty_node_list2(list_degree[-10:]))
#			print("Degree Centrality", TopicMiner.pretty_node_list2(list_degree_centrality[-10:]))
#			print("Eigenvector Centrality", TopicMiner.pretty_node_list2(list_eigenvector_centrality[-10:]))
#			print("Clustering", TopicMiner.pretty_node_list2(list_clustering[-10:]))



#			print("KEYWORDS SUMMARY:", summary)


			importance = sum([map_node_centrality[n] for n in summary])

			current_topic = Topic(list(summary), importance, measures=measures)
			results.append(current_topic)
			
	#		print("MOST CONNECTED nodes:", pretty_node_list(filter_most_degree_nodes(subgraph), 0.8, 10))
	#		print("MOST CENTRAL nodes", pretty_node_list(filter_most_central_nodes(subgraph), 0.8, 10))

			if rec_level > 0:

				subgraph = graph.subgraph(list_nodes)

				subpartition = community_louvain.best_partition(subgraph, resolution=resolution*2.)

				list_subcom = []

				for subcom in set(partition.values()) :
					sublist_nodes = [node for node in subpartition.keys() if subpartition[node] == subcom]
					if not sublist_nodes:
						continue

					importance = sum([map_node_centrality[n] for n in sublist_nodes])

					list_subcom.append([sublist_nodes, importance])

				list_subcom = sorted(list_subcom, key=lambda x:x[1], reverse=True)

				list_subtopics = []

				for sublist_nodes, importance in list_subcom:
	#				print("     === score:", ("%.2f" % importance))
	#				print("      ", sublist_nodes)				
					list_subtopics.append(Topic(list(sublist_nodes), importance))

	#			print(list_subtopics)

				current_topic.subtopics = list_subtopics

		return results


"""
Visualization
"""

# TODO

class TopicVisualizer:

	def __init__(self):
		pass

	@staticmethod
	def prettyPrintTopicsList(list_topics, importance_threshold=0, length_threshold=0, rec_level=0, max_rec=None, kw_sep=" ", print_all=False, fd=None, oneline_output=True):

		if fd is None:
			fd = sys.stdout

		if max_rec is not None and max_rec == rec_level:
			return

		prefix = "   "*rec_level

		max_importance = max([t.importance for t in list_topics]) if list_topics else 1.
		#min_importance = min([t.importance for t in list_topics])

		list_topics = sorted(list_topics, key=lambda x: x.importance, reverse=True)

		for idx, topic in enumerate(list_topics):

			if not print_all:
				if topic.importance < importance_threshold*max_importance :
					continue

				if len(topic.keywords) < length_threshold:
					continue

			print("", file=fd)
			print(prefix, "=== ", ("sub"*rec_level) + "topic", idx, file=fd)
			print(prefix, "aggregated centrality:", "%.2f" % topic.importance, "count items:", len(topic.keywords), "count subtopics:", len(topic.subtopics), file=fd)
			if oneline_output:
				print(prefix, "topic:", kw_sep.join(topic.keywords), idx, file=fd)
			else:
				for kw in topic.keywords:
					print(prefix + "   ", kw, file=fd)
			if len(topic.subtopics) > 1:
				print(prefix, "subtopics:", idx, file=fd)
				TopicVisualizer.prettyPrintTopicsList(topic.subtopics, importance_threshold, length_threshold, rec_level + 1, max_rec, kw_sep, print_all, fd, oneline_output)


