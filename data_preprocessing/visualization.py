import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.colors as colors
import matplotlib.cm as cmx


def visualize_graph(graph_data, labels=False, node_size=6, line_width=5, save=False):
    """
    Plot pytorch geometric graph object.
    :param graph_data: pytorch geometric graph object
    :param labels: Whether to plot with label information (colored nodes)
    :param node_size: size of nodes in graph
    :param line_width: strength of edges between nodes in graph
    :return: nothing (plots graph)
    """
    graph_viz = to_networkx(graph_data)
    if labels:
        node_labels = graph_data.y[list(graph_viz.nodes)].numpy()
        ColorLegend = {'News Text': 0, 'Tweets/Retweets': 1, 'Users': 2}
        cNorm = colors.Normalize(vmin=0, vmax=2)
        jet = cm = plt.get_cmap('brg')
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        f = plt.figure(1)
        ax = f.add_subplot(1, 1, 1)
        for label in ColorLegend:
            ax.plot([0], [0], color=scalarMap.to_rgba(ColorLegend[label]), label=label, marker='.', linestyle='None')
        nx.draw(graph_viz, cmap=jet, vmin=0, vmax=2, arrowstyle='-', node_color=node_labels, width=0.3, node_size=node_size,
                linewidths=line_width, ax=ax)
        plt.axis('off')
        f.set_facecolor('w')
        plt.legend()
        f.tight_layout()
        if save:
            f.savefig('graph.eps', format='eps')
    else:
        plt.figure(1, figsize=(7, 7))
        nx.draw(graph_viz, cmap=plt.get_cmap('Set1'), arrowstyle='-', node_size=node_size, linewidths=line_width)
    plt.show()
