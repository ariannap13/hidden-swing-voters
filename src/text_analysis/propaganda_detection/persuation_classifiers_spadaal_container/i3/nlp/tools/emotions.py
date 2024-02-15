from typing import List
from collections import Counter
import numpy as np
import networkx as nx
import pandas as pd

"""
Computations
"""


class EmotionAnalyzer:

	list_emotions = ["sad", "anger", "joy", "disgust", "fear", "surprise", "neutral"]
	list_sentiment = ["positive", "neutral", "negative"]

	def __init__(self):
		pass

	@staticmethod
	def compute_histogram_word_emotions(corpus):
		""" compute the histogram of words associated with emotion and networks
		
		corpus: TextCorpus object
		"""

		list_sentiment = EmotionAnalyzer.list_sentiment
		list_emotions = EmotionAnalyzer.list_emotions

		## compute counts

		map_emo_count = {}
		for emo in list_emotions:
			map_emo_count[emo] = Counter()

		map_sent_count = {}
		for sent in list_sentiment:
			map_sent_count[sent] = Counter()

		for doc in corpus.documents:
			emo = doc.tags["emotion"]
			if emo is not None:
				map_emo_count[emo].update(doc.tokens)

			sent = doc.tags["sentiment"]
			if sent is not None:
				map_sent_count[sent].update(doc.tokens)

		## save as a CSV file

		list_words = set()
		for emo in list_emotions:
			words = map_emo_count[emo].keys()
			list_words.update(set(words))

		for sent in list_sentiment:
			words = map_sent_count[sent].keys()
			list_words.update(set(words))

		## create dataframe word emotions

		labels = ("word,"+",".join(list_emotions)+",tot_emo,"+",".join(list_sentiment)+",tot_sent").split(",")

		all_lines = []

		for word in list_words:
			line = []
			line.append(word)

			# process emotions
			sum_emo = 0
			for emo in list_emotions:
				count = map_emo_count[emo][word] if word in map_emo_count[emo] else 0
				line.append(count)
				sum_emo += count
			line.append(sum_emo)

			# process sentiments
			sum_sent = 0
			for sent in list_sentiment:
				count = map_sent_count[sent][word] if word in map_sent_count[sent] else 0
				line.append(count)
				sum_sent += count
			line.append(sum_sent)

			# add to dataframe

			all_lines.append(line)

		df_word_emotions = pd.DataFrame.from_records(all_lines, columns=labels)

		print(df_word_emotions)

		## build entity-emotion graph

		graph_emo = nx.Graph()

		for emo in list_emotions:
			for w, count in map_emo_count[emo].most_common():
				if count > 10:
					graph_emo.add_edge(emo, w, weight=count)

		#nx.write_gexf(graph_emo, output_dir+"graph-ent_emo.gexf")

		graph_sent = nx.Graph()

		for sent in list_sentiment:
			for w, count in map_sent_count[sent].most_common():
				if count > 10:
					graph_sent.add_edge(sent, w, weight=count)

		#nx.write_gexf(graph_sent, output_dir+"graph-ent_sent.gexf")

		ret_data = {'graph_sent': graph_sent,
					'graph_emo': graph_emo,
					'map_sent_count': map_sent_count,
					'map_emo_count': map_emo_count,
					'df_word_emotions': df_word_emotions}

		return ret_data

	@staticmethod
	def saveAllDataWordHistogram(data, output_dir):
		""" save all the data related to word hisgoram emotion to files
		
		output_dir: destination folder for the files
		"""

		graph_emo = data["graph_emo"]
		graph_sent = data["graph_sent"]
		map_emo_count = data["map_emo_count"]
		map_sent_count = data["map_sent_count"]
		df_word_emotions = data["df_word_emotions"]

		## save as independant word histograms

		for emo in EmotionAnalyzer.list_emotions:
			print("== emo", emo)
			print(map_emo_count[emo].most_common(20))
			np.savetxt(output_dir+"histo_emo_"+emo+".txt", np.array(map_emo_count[emo].most_common()), fmt="%s")

		for sent in EmotionAnalyzer.list_sentiment:
			print("== sent", sent)
			print(map_sent_count[sent].most_common(20))
			np.savetxt(output_dir+"histo_sent_"+sent+".txt", np.array(map_sent_count[sent].most_common()), fmt="%s")


		df_word_emotions.to_csv(output_dir+"word_emotions.csv")

		nx.write_gexf(graph_emo, output_dir+"graph-ent_emo.gexf")
		nx.write_gexf(graph_sent, output_dir+"graph-ent_sent.gexf")


"""
Visualization
"""

import sys
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

class EmotionVisualizer:

	def __init__(self):
		pass

	@staticmethod
	def plotRadarWord(word, df, save_to_file=None, show=True):
		""" plot a radar plot for a word

		df: word emotion dataframe
		save_to_file: if not None, save to disk
		show: if True if display the plot
		"""

		categories=["sad",  "anger",  "joy",  "disgust",  "fear",  "surprise"]
		N = len(categories)
		
		df_word=df[df["word"] == word]
		values = df_word[categories].to_numpy(dtype=float)[0]
		tot_emo = df_word[["tot_emo"]].to_numpy(dtype=float)[0][0]
		values /= tot_emo
		values = values.tolist()
		values += values[:1] # repeat first value to close circular graph

		angles = [n / float(N) * 2 * pi for n in range(N)]
		angles += angles[:1]
		
		ax = plt.subplot(111, polar=True)
		
		plt.xticks(angles[:-1], categories, color='grey', size=8)
		
		ax.set_rlabel_position(0)
		plt.yticks([0., 0.25, 0.5, 0.75, 1.], ["0","0.25","0.5", "0.75", "1."], color="grey", size=7)
		plt.ylim(0,1.)
		
		ax.plot(angles, values, linewidth=1, linestyle='solid')
		
		ax.fill(angles, values, 'b', alpha=0.1)

		plt.title("Emotion Radar plot of word: "+word)

		if show:
			plt.show()

		if save_to_file:
			plt.savefig(save_to_file)