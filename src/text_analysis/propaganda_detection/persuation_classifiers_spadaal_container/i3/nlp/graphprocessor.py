

from tqdm import tqdm 

import networkx as nx

import typing

from i3.nlp.corpus import TextCorpus


class GraphProcessor:

	def __init__(self):
		pass

	@staticmethod
	def getNodeBlanket(g, node):
		return list(g.neighbors(node))

	@staticmethod
	def getNodeListBlanket(g, list_node):
		ret = []
		for n in list_node:
			if n not in g.nodes():
				continue
			ret.extend(n)
			ret.extend(GraphProcessor.getNodeBlanket(g, n))
		return nx.Graph(g.subgraph(ret))

	@staticmethod
	def getNodesSortedByCentrality(graph):

		map_centrality = nx.degree_centrality(graph)
		nodes_centrality = sorted(map_centrality.items(), key=lambda x: x[1], reverse=True)

	#	final_nodes = [ (x,v) for x,v in nodes_centrality if v >= nodes_centrality[int(len(nodes_centrality) * prop)][1]]

		return nodes_centrality

	@staticmethod
	def getNodesSortedByDegree(graph):
		# prop: the degre must be igher thant prop% of the nodes
	#	final_nodes = graph.nodes

		sorted_degrees = sorted([(x, graph.degree(x)) for x in graph.nodes], key=lambda x:x[1])

	#	final_nodes = [(x,v) for x,v in sorted_degrees if v >= sorted_degrees[int(len(sorted_degrees) * prop)][1]]

		return sorted_degrees

	@staticmethod
	def filterNodeDegree(graph, min_degree, replace=True):
		""" filter nodes based on their weight """

		if not replace:
			graph = nx.Graph(graph)

		list_toremove = [x for x in graph.nodes if graph.degree(x) < min_degree]
		for n in list_toremove:
			graph.remove_node(n)

		return graph


	@staticmethod
	def filterEdgeWeight(graph, min_edge_weight, replace=True):
		""" filter edges based on their weight """

		list_toremove = []
		for u,v,a in graph.edges(data=True):
			w = a["weight"]
			if w < min_edge_weight:
				list_toremove.append((u,v))

		if not replace:
			graph = nx.Graph(graph)

		for u,v in list_toremove:
			graph.remove_edge(u, v)

		return graph

	@staticmethod
	def buildGraphFromDocuments(documents, graph=None, ignore_nodes=[], linking_dist=1, min_degree=None, min_weight=None):
		""" build word graph linking words within a radius of n words in documents """

		ignore_nodes = set(ignore_nodes)

		graph = nx.Graph() if graph is None else graph

		for doc in tqdm(documents):
			keywords = doc if type(doc) == list else doc.text.split()
			for i in range(len(keywords)):
				label1 = keywords[i]
				if label1 in ignore_nodes:
					continue
				for j in range(max(0, i - linking_dist), min(len(keywords), i + linking_dist)):
					if j < 0 or j >= len(keywords) or i < j or i == j:
						continue
					label2 = keywords[j]
					if label2 in ignore_nodes:
						continue
					if not graph.has_edge(label1, label2):
						graph.add_edge(label1, label2, weight=1)
					else:
						w = graph[label1][label2]["weight"]
						graph.add_edge(label1, label2, weight=w+1)
		
		if min_degree is not None:
			GraphProcessor.filterNodeDegree(graph, min_degree)

		if min_weight is not None:
			GraphProcessor.filterEdgeWeight(graph, min_weight)

		return graph

	@staticmethod
	def processCorpus(corpus: TextCorpus, ignore_nodes=[], linking_dist=1, min_degree=None, min_weight=None, processTokens=True):

		docs = corpus.getDocumentsTokens() if processTokens else corpus.getDocumentsTexts()

		print("* build graph")

		return GraphProcessor.buildGraphFromDocuments(docs, ignore_nodes=ignore_nodes, linking_dist=linking_dist, min_degree=min_degree, min_weight=min_weight)	