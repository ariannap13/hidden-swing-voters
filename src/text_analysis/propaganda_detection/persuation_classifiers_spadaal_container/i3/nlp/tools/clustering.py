from collections import Counter
import numpy as np
from scipy import spatial

"""
def summarise_clusters_proximity_to_centroids(labels, all_txt, all_emb, all_lang, top_n=2, verbose=False):


#	print("XXX", top_n, len(all_txt), int(len(all_txt)/3.))

	top_n_default = top_n

	map_cl_data = {}

	centroids = []

	for cl_id, txt, emb, lang in zip(labels, all_txt, all_emb, all_lang):

		if cl_id not in map_cl_data:
			map_cl_data[cl_id] = []
		map_cl_data[cl_id].append([txt, emb, lang])

#		print(lang, txt)

	for cl_id, data in map_cl_data.items():
		cl_embs = [x[1] for x in data]

		avg_emb = np.array(cl_embs).mean(axis=0)
		centroids.append(avg_emb)

		for d in data:
			dist = 1. - spatial.distance.cosine(avg_emb, d[1])

			d.append(dist)

	#	print(map_cl_data)

	list_clusters = []
	i=0

	for cl_id, data in sorted(map_cl_data.items(), key=lambda x: len(x[1]), reverse=True):

		data = sorted(data, key=lambda x: x[3], reverse=True)

		top_n = max(1 if len(data) < 6 else 2, min(top_n, round(len(data)/3.)))

		count_lang = Counter([x[2] for x in data])

		if verbose:
			print("* CLUSTER >>", cl_id, "<<", "len:", len(data), count_lang)

		clust_sum = { "id": cl_id, "count": len(data), "count_lang": count_lang.most_common(), "top": [], "mid": [], "low": [], "top_by_lang": {}, "all": [], "centroid_vector": centroids[i]}
		i=i+1
		list_clusters.append(clust_sum)

		if verbose:
			print("")
			print("top ----")


		for txt, emb, lang, dist in data[:top_n]:
			if verbose:
				print("   ", "%.2f" % dist, "-", txt)
				print("")


			clust_sum["top"].append([txt, dist])

		if verbose:
			print("")
			print("mid ----")

		half_top_n = int(top_n / 2.) if top_n > 2 else 1

		for txt, emb, lang, dist in data[int(len(data)/2.)-half_top_n:int(len(data)/2.)+half_top_n]:
			if verbose:
				print("   ", "%.2f" % dist, "-", txt)
				print("")

			clust_sum["mid"].append([txt, dist])

		if verbose:
			print("")
			print("low ----")

		for txt, emb, lang, dist in data[-top_n:]:
			if verbose:
				print("   ", "%.2f" % dist, "-", txt)
				print("")

			clust_sum["low"].append([txt, dist])


		if verbose:
			print("")
			print("top by language ---- ----")

		for lang in count_lang.keys():
			data_lang = [x for x in data if x[2] == lang]

			for txt, emb, _, dist in data_lang[:2]:
				if verbose:
					print("   ", lang, ":", "%.2f" % dist, "-", txt)
					print("")

				if lang not in clust_sum["top_by_lang"]:
					clust_sum["top_by_lang"][lang] = []
				clust_sum["top_by_lang"][lang].append([txt, dist])

			if verbose:	
				print("    --")


		if verbose:
			print("all ---- ---- ----")

		for txt, emb, lang, dist in data:
			print("   ", "%.4f" % dist, "-", txt)

			clust_sum["all"].append([txt, dist, lang])


		print("")
		print("")

	return list_clusters
"""

def summarise_clusters_proximity_to_centroids(labels, all_txt, all_emb, all_lang, top_n=2, verbose=False):

	map_cl_data = {}

	centroids = []

	for cl_id, txt, emb, lang in zip(labels, all_txt, all_emb, all_lang):

		if cl_id not in map_cl_data:
			map_cl_data[cl_id] = []
		map_cl_data[cl_id].append([txt, emb, lang])

	for cl_id, data in map_cl_data.items():
		cl_embs = [x[1] for x in data]		

		avg_emb = np.array(cl_embs).mean(axis=0)
		centroids.append(avg_emb)

		for d in data:
			dist = 1. - spatial.distance.cosine(avg_emb, d[1])

			d.append(dist)

	list_clusters = []
	i=0

	for cl_id, data in map_cl_data.items():

		data = sorted(data, key=lambda x: x[3], reverse=True)
		
		count_lang = Counter([x[2] for x in data])

		if verbose:
			print("* CLUSTER >>", cl_id, "<<", "len:", len(data), count_lang)

		clust_sum = { "id": cl_id, "count": len(data), "count_lang": count_lang.most_common(), "top": [], "mid": [], "low": [], "top_by_lang": {}, "all": [], "centroid_vector": centroids[i]}
		i=i+1		

		if verbose:
			print("top")

		for txt, emb, lang, dist in data[:top_n]:
			if verbose:
				print("   ", "%.2f" % dist, "-", txt)

			clust_sum["top"].append([txt, dist])

		if verbose:
			print("mid")

		half_top_n = int(top_n / 2.) if top_n > 2 else 1

		for txt, emb, lang, dist in data[int(len(data)/2.)-half_top_n:int(len(data)/2.)+half_top_n]:
			if verbose:
				print("   ", "%.2f" % dist, "-", txt)

			clust_sum["mid"].append([txt, dist])


		if verbose:
			print("low")

		for txt, emb, lang, dist in data[-top_n:]:
			if verbose:
				print("   ", "%.2f" % dist, "-", txt)

			clust_sum["low"].append([txt, dist])


		if verbose:
			print("    ------")

		for lang in count_lang.keys():
			data_lang = [x for x in data if x[2] == lang]

			for txt, emb, _, dist in data_lang[:2]:
				if verbose:
					print("   ", lang, ":", "%.2f" % dist, "-", txt)

				if lang not in clust_sum["top_by_lang"]:
					clust_sum["top_by_lang"][lang] = []
				clust_sum["top_by_lang"][lang].append([txt, dist])
		
		if verbose:
			print("    ------")

		for txt, emb, lang, dist in data:
			clust_sum["all"].append([txt, dist, lang])

		list_clusters.append(clust_sum)		

	return list_clusters

if __name__ == "__main__":

	labels = [1, 1, 1, 1, 1]
	all_txt = ["a", "b", "c", "d", "e"]
	all_emb = [ np.random.rand(3) for i in range(5)]
	all_lang = ["en", "fr", "en", "es", "es"]

	print(summarise_clusters_proximity_to_centroids(labels, all_txt, all_emb, all_lang))