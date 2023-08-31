# Libraries
import pandas as pd
import networkx as nx
import argparse

#relevant_parties = ["Alleanza Verdi Sinistra", "Azione - Italia Viva",
#                    "Fratelli d'Italia", "Lega", "Movimento 5s", "Partito Democratico"]


def prepare_representatives(path_representative):
    print("-"*50)
    print("Preparing representatives...")
    print()

    # Reading representatives metadata
    representatives = pd.read_csv(path_representative, usecols=(range(5, 10)))
    print("Representative has shape: ", representatives.shape)
    print(representatives.head())
    print()

    # Filtering representatives
    # print("Relevant parties: ", relevant_parties)
    # representatives = representatives[representatives["Party"].isin(relevant_parties)]

    # Removing unwanted data
    representatives = representatives[representatives.ids != '-----']
    representatives = representatives[representatives.ids != 'not found']
    print("(Clean) Representative has shape: ", representatives.shape)
    print(representatives.head())
    print()

    # Keeping only representative ids
    representative_users = representatives['ids'].unique()
    print("Representative users: ", len(representative_users))
    print(representative_users[:10])

    return representative_users


def parse_data(path_raw_data):
    print("-"*50)
    print("Reading raw data...")
    print()

    # Reading data
    data = pd.read_csv(path_raw_data, dtype=str)
    print("Raw data has shape: ", data.shape)
    print(data.head())
    print()

    # Dropping unwanted columns
    print("Dropping unwanted columns...")
    data = data.drop(columns=['text_tweet_id', 'created_at', 'type'])

    # Merging by (source, dest) pairs and counting
    print("Merging by (source, dest) pairs and counting...")
    data = data.groupby(['source', 'dest']).size().reset_index(name='weight').sort_values(by=['weight'], ascending=False)
    print("Merged data has shape: ", data.shape)
    print(data.head())
    print()

    return data


def save_graph(g, out_path):
    print("-"*50)
    print("Saving graph...")
    print()

    # Saving graph
    g_data = nx.to_pandas_edgelist(g)
    g_data.to_csv(out_path, index=False)
    print("Graph saved!")
    print()


def create_graph(data):
    print("-"*50)
    print("Creating graph...")
    print()

    # Creating graph
    graph = nx.from_pandas_edgelist(data,
                                    source='source',
                                    target='dest',
                                    edge_attr='weight',
                                    create_using=nx.DiGraph())

    # Some basic stats
    print("Number of nodes: ", graph.number_of_nodes())
    print("Number of edges: ", graph.number_of_edges())
    print("Raw edges: ", data.shape[0])
    print()

    return graph


def create_representative_graph(data, representative_users):
    print("-"*50)
    print("Creating representative graph...")
    print()

    # Filtering edges containing representative users
    print("Filtering edges containing representative users...")
    data = data[data['source'].isin(representative_users) | data['dest'].isin(representative_users)]

    # Creating graph
    graph = nx.from_pandas_edgelist(data,
                                    source='source',
                                    target='dest',
                                    edge_attr='weight',
                                    create_using=nx.DiGraph())
    
    # Some basic stats
    print("Number of nodes: ", graph.number_of_nodes())
    print("Number of edges: ", graph.number_of_edges())
    print("Raw edges: ", data.shape[0])
    print()

    return graph


def parse_args():
    parser = argparse.ArgumentParser(description='Election ITA Networks Modeling')
    parser.add_argument('--path-raw-data', type=str, help='Path to raw data')
    parser.add_argument('--path-representative', type=str, default='../data/raw_data/twitter_representatives_handles_final.csv', help='Path to representative data')
    parser.add_argument('--out-graph', help='Path to store the graph')
    parser.add_argument('--out-graph-representative', help='Path to store the representative graph')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Due to the large volume of data (around 19 million tweets), 
    # we prune the retweet networks by keeping only those edges (i.e., retweets) 
    # involving at least a political representative.

    # Preparing representative users
    representative_users = prepare_representatives(args.path_representative)

    # Parsing raw data
    data = parse_data(args.path_raw_data)

    # Creating graph
    graph = create_graph(data)

    # Creating representative graph
    graph_representative = create_representative_graph(data, representative_users)

    # Saving graphs
    save_graph(graph, args.out_graph)
    save_graph(graph_representative, args.out_graph_representative)


if __name__ == "__main__":
    main()