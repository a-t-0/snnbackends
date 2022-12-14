"""File used to generate graph plots."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import networkx as nx
from typeguard import TYPE_CHECKING, typechecked

if TYPE_CHECKING:
    from snncompare.tests.test_scope import Long_scope_of_tests


@typechecked
def plot_circular_graph(
    density: float,
    G: nx.DiGraph,
    recurrent_edge_density: int | float,
    test_scope: Long_scope_of_tests,
) -> None:
    """Generates a circular plot of a (directed) graph.

    :param density: param G:
    :param seed: The value of the random seed used for this test.
    :param export:  (Default value = True)
    :param show: Default value = True)
    :param G: The original graph on which the MDSA algorithm is ran.
    :param recurrent_edge_density:
    :param test_scope:
    """
    # the_labels = get_alipour_labels(G, configuration=configuration)
    the_labels = get_labels(G, "du")
    # nx.draw_networkx_labels(G, pos=None, labels=the_labels)
    npos = nx.circular_layout(
        G,
        scale=1,
    )
    nx.draw(G, npos, labels=the_labels, with_labels=True)
    if test_scope.export:

        create_target_dir_if_not_exists("latex/Images/", "graphs")
        plt.savefig(
            f"latex/Images/graphs/graph_{test_scope.seed}_size{len(G)}_"
            + f"p{density}_p_recur{recurrent_edge_density}.png",
            dpi=200,
        )
    if test_scope.show:
        plt.show()
    plt.clf()
    plt.close()


@typechecked
def plot_uncoordinated_graph(
    G: nx.DiGraph | nx.Graph, filepath: str = None, show: bool = True
) -> None:
    """Generates a circular plot of a (directed) graph.

    :param density: param G:
    :param seed: The value of the random seed used for this test.
    :param show: Default value = True)
    :param G: The original graph on which the MDSA algorithm is ran.
    :param export:  (Default value = False)
    """
    # TODO: Remove unused method.
    # the_labels = get_alipour_labels(G, configuration=configuration)
    # the_labels =
    # nx.draw_networkx_labels(G, pos=None, labels=the_labels)
    npos = nx.circular_layout(
        G,
        scale=1,
    )
    nx.draw(G, npos, with_labels=True)
    if isinstance(filepath, str):
        create_target_dir_if_not_exists("latex/Images/", "graphs")
        plt.savefig(
            filepath,
            dpi=200,
        )
    if show:
        plt.show()
    plt.clf()
    plt.close()


@typechecked
def create_target_dir_if_not_exists(path: str, new_dir_name: str) -> None:
    """Creates an output dir for graph plots.

    :param path: param new_dir_name:
    :param new_dir_name:
    """

    create_root_dir_if_not_exists(path)
    if not os.path.exists(f"{path}/{new_dir_name}"):
        os.makedirs(f"{path}/{new_dir_name}")


@typechecked
def create_root_dir_if_not_exists(root_dir_name: str) -> None:
    """

    :param root_dir_name:

    """
    if not os.path.exists(root_dir_name):
        os.makedirs(f"{root_dir_name}")
    if not os.path.exists(root_dir_name):
        raise Exception(f"Error, root_dir_name={root_dir_name} did not exist.")


@typechecked
def get_labels(G: nx.DiGraph, configuration: str) -> dict[int, str]:
    """Returns the labels for the plot nodes.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param configuration:
    """
    labels = {}
    for node_name in G.nodes:
        if configuration == "du":
            labels[node_name] = f"{node_name}"
            # ] = f'{node_name},R:{G.nodes[node_name]["lava_LIF"].du.get()}'
    return labels
