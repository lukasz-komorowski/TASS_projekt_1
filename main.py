import networkx as nx
from random import sample
import time
import matplotlib.pyplot as plt
import powerlaw
import numpy as np


def average_path(number, graph):
    average_path_length = 0
    start_time = time.time()
    for x in range(number):
        sample_nodes = sample(list(nx.nodes(graph)), 2)
        shortest_path = nx.shortest_path_length(graph, source=sample_nodes[0], target=sample_nodes[1])
        average_path_length += shortest_path
    print("--- %s seconds ---" % (time.time() - start_time))
    return average_path_length/number


start_time = time.time()

g = nx.read_edgelist("zadB_1.txt", create_using=nx.MultiGraph())

print("Given graph:")
print(f"Number of edges: {nx.number_of_edges(g)}")
print(f"Number of nodes: {nx.number_of_nodes(g)}")
print(f"Number of selfloops: {nx.number_of_selfloops(g)}\n")

g.remove_edges_from(list(nx.selfloop_edges(g)))
g = nx.Graph(g)
print("Graph after removing loops and parallel edges:")
print(f"Number of edges: {nx.number_of_edges(g)}")
print(f"Number of nodes: {nx.number_of_nodes(g)}")
print(f"Number of selfloops: {nx.number_of_selfloops(g)}\n")

Gcc = max(nx.connected_components(g), key=len)
g = g.subgraph(Gcc)
print("Biggest component:")
print(f"Number of edges: {nx.number_of_edges(g)}")
print(f"Number of nodes: {nx.number_of_nodes(g)}")
print(f"Number of selfloops: {nx.number_of_selfloops(g)}\n")

# --------------------- średnie ścieżka - 100, 1000, 10000 ---------------------
print(f"Average path for 100 random pairs of vertices: {average_path(100, g)}")
print(f"Average path for 1000 random pairs of vertices: {average_path(1000, g)}")
print(f"Average path for 10 000 random pairs of vertices: {average_path(10000, g)}\n") # uwaga - dlugo sie liczy

# --------------------- rzędy rdzeni ---------------------
cores = nx.core_number(g)
high_1 = max(cores.values())
high_2 = 0
high_3 = 0
for i in cores.values():
    if i > high_2 and i != high_1:
        high_2 = i
    if i > high_3 and i != high_2 and i != high_1:
        high_3 = i
print(f"Highes values for core k: {high_1}, {high_2}, {high_3}")
k_core = nx.k_core(g)
k_core_1 = nx.k_core(g, k=high_1)
k_core_2 = nx.k_core(g, k=high_2)
k_core_3 = nx.k_core(g, k=high_3)
print(f"Number of cores k = {high_1}: {nx.number_connected_components(k_core_1)}")
print(nx.info(k_core_1))
print(f"\nNumber of cores k = {high_2}: {nx.number_connected_components(k_core_2)}")
print(nx.info(k_core_2))
print(f"\nNumber of cores k = {high_3}: {nx.number_connected_components(k_core_3)}")
print(nx.info(k_core_3))

# --------------------- rozkład wierzchołków ---------------------
degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())
plt.bar(deg, cnt, color="b")
plt.title("Rozkład stopni wierzchołków - histogram")
plt.ylabel("Liczba")
plt.xlabel("Stopień")
plt.show()


# --------------------- WYKRES HILLA ---------------------
dlist = [d for n, d in g.degree()]
NBINS = 150
bins = np.logspace(np.log10(min(dlist)), np.log10(max(dlist)), num=NBINS)
bcnt, bedge = np.histogram(np.array(dlist), bins=bins)
alpha = np.zeros(len(bedge[:-2]))

for i in range(len(bedge) - 2):
    fit = powerlaw.Fit(dlist, xmin=bedge[i], discrete=True)
    alpha[i] = fit.alpha

fig, ax = plt.subplots()
ax.semilogx(bedge[:-2], alpha)
ax.set_title('Wykres Hilla')
plt.show()

# --------------------- Wykładnik ---------------------
fit = powerlaw.Fit(dlist, discrete=True)
print(f'alpha =  {str(fit.alpha)}')


print("\n--- %s seconds ---" % (time.time() - start_time))
