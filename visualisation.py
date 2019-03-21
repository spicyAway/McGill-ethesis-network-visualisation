import numpy as np
import pandas as pd
import csv
import math
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, StaticLayoutProvider, LinearColorMapper, ColumnDataSource
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.palettes import Spectral8, Spectral4
import networkx as nx
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('graph.csv')
node_indices = df['Target']

# Map cubehelix_palette
palette = sns.cubehelix_palette(21)
pal_hex_lst = palette.as_hex()

#
# plot = figure(title='Graph Layout Demonstration', x_range=(-1.1,1.1), y_range=(-1.1,1.1),
#               tools='', toolbar_location=None)
#
# graph = GraphRenderer()
# graph.node_renderer.data_source.add(node_indices, 'index')
# graph.node_renderer.glyph = Oval(height=0.1, width=0.2, fill_color='color')

# start_nodes = df['Source'].tolist()
# end_nodes = df['Target'].tolist()

# graph.edge_renderer.data_source.data = dict(
#     start=start_nodes,
#     end=end_nodes)

# edges = list(zip(start_nodes, end_nodes))
# G=nx.Graph()
# G.add_nodes_from(node_indices.tolist())
# G.add_edges_from(edges)

G = nx.from_pandas_edgelist(df, source = 'Source',
                                target = 'Target', edge_attr = 'Score')

print(type(G.degree()))
node_size = {}
for node in G.nodes:
    v = G.degree[node]
    node_size[node] = 5*v
# node_size = {k:5*v for k,v in G.degree()}
# Some Random index
node_color = {k:v for k,v in enumerate(np.random.uniform(low=0, high=21, size=(G.number_of_nodes(),)).round(1))}

### set node attributes
nx.set_node_attributes(G,node_color, 'node_color')
nx.set_node_attributes(G,node_size,'node_size')

source=ColumnDataSource(pd.DataFrame.from_dict({k:v for k,v in G.nodes(data=True)},orient='index'))
mapper = LinearColorMapper(palette=pal_hex_lst, low=0, high=21)

### Initiate bokeh plot
plot = figure(title="Resized Node Demo", x_range=(-1.1,1.1), y_range=(-1.1,1.1),
          tools="", toolbar_location=None)

# Graph renderer using nx
graph = from_networkx(G, nx.spring_layout, scale=2, center=(0,0))

# Style node
graph.node_renderer.data_source = source
graph.node_renderer.glyph = Circle(size='node_size', fill_color={'field': 'node_color', 'transform': mapper})
plot.renderers.append(graph)

show(plot)

# plt.figure(figsize = (10,9))
# nx.draw_networkx(graph)
# plt.show()
