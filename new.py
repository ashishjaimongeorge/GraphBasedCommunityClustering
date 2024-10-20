import pandas as pd
from community import community_louvain
from sentence_transformers import SentenceTransformer
import igraph as ig
import leidenalg
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import argparse
import os
from FlagEmbedding import FlagModel

model = FlagModel('BAAI/bge-large-en-v1.5', use_fp16=False)
bert_model = SentenceTransformer("bert-large-nli-stsb-mean-tokens", device='cpu')
roberta_model = SentenceTransformer("roberta-large-nli-stsb-mean-tokens", device='cpu')

input_path = "Input.csv"
threshold = 0.83

raw_asp_df = pd.read_csv(input_path)
raw_asps = raw_asp_df[~raw_asp_df["Actionables"].isna()]["Actionables"].values

raw_asp_vectors_flag = model.encode(raw_asps)
print("Flag embedded")
raw_asp_vectors_bert = bert_model.encode(raw_asps, show_progress_bar=True)
print("BERT embedded")
raw_asp_vectors_roberta = roberta_model.encode(raw_asps, show_progress_bar=True)
print("RoBERTa embedded")

def l2_normalize_rows(array):
    l2_norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.where(l2_norms == 0, 1, l2_norms)


normalized_bert = l2_normalize_rows(raw_asp_vectors_bert)
normalized_roberta = l2_normalize_rows(raw_asp_vectors_roberta)
normalized_flag = l2_normalize_rows(raw_asp_vectors_flag)

combined_vectors = np.maximum.reduce([normalized_bert, normalized_roberta, normalized_flag])
cos_grid_use = cosine_similarity(combined_vectors)


G_nx = nx.Graph()
G_nx.add_nodes_from(raw_asps)

for i in tqdm(range(len(raw_asps))):
    for j in range(i, len(raw_asps)):
        similarity = cos_grid_use[i][j]
        if similarity>threshold:
            G_nx.add_edge(raw_asps[i], raw_asps[j], weight=(similarity-threshold)/(1-threshold))

resolution = 1.0

random_state = 44

partition = community_louvain.best_partition(G_nx, resolution=resolution, random_state=random_state)

modularity_score = community_louvain.modularity(partition, G_nx)
print("Modularity of the partition:", modularity_score)

communities = []
for com in set(partition.values()):
    list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
    communities.append(list_nodes) 

communities_out = pd.DataFrame()
count_more_than_one = 0

for i, c in enumerate(communities):
    if len(c) >= 1:
        df_index = len(communities_out)
        communities_out.loc[df_index, 'Context aspects'] = '|'.join(c) if len(c) > 1 else c[0]
        communities_out.loc[df_index, 'Aspect'] = c[0]
        communities_out.loc[df_index, 'CA Count'] = len(c)
        if len(c) > 1:
            count_more_than_one += 1

communities_out_sorted = communities_out.sort_values(by='Count', ascending=False)

print(f"Groups with more than one member: {count_more_than_one}")

communities_out_sorted.to_csv("Output.csv",index = False)



