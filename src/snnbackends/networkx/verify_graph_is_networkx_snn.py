"""Verifies the graph represents a connected and valid SNN, with all required
neuron and synapse properties specified."""

# Import the networkx module.
from typing import Tuple

import networkx as nx
from networkx.classes.digraph import DiGraph
from typeguard import typechecked

from snnbackends.networkx.LIF_neuron import Synapse


@typechecked
def assert_synaptic_edgeweight_type_is_correct(
    G: nx.DiGraph, edge: nx.DiGraph.edges
) -> None:
    """

    :param edge: nx.DiGraph.edges:
    :param G: The original graph on which the MDSA algorithm is ran.
    :param edge: nx.DiGraph.edges:

    """
    if nx.get_edge_attributes(G, "weight") != {}:

        if not isinstance(G.edges[edge]["weight"], Synapse):
            raise Exception(
                f"Weight of edge {edge} is not a"
                + " Synapse object. It is"
                + f': {G.edges[edge]["weight"]} of type:'
                f'{type(G.edges[edge]["weight"])}'
            )
    else:
        raise Exception(
            f"Weight of edge {edge} is an attribute (in the"
            + ' form of: "weight").'
        )


@typechecked
def assert_synapse_properties_are_specified(
    snn_graph: DiGraph, edge: Tuple[str, str]
) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param edge:

    """
    if not has_valid_synapse(snn_graph, edge):
        raise Exception(
            f"Not all synapse properties of edge: {edge} are"
            + " specified (correctly): "
            + f"{snn_graph.edges[edge]}"
        )


@typechecked
def has_valid_synapse(snn_graph: DiGraph, edge: Tuple[str, str]) -> bool:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param edge:

    """
    for key in snn_graph.edges[edge].keys():
        if key not in ["synapse", "is_redundant"]:
            return False
        if key == "synapse":
            if not isinstance(snn_graph.edges[edge]["synapse"], Synapse):
                return False
    return True


@typechecked
def assert_no_duplicate_edges_exist(G: DiGraph) -> None:
    """Asserts no duplicate edges exist, throws error otherwise.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    visited_edges = []
    for edge in G.edges:
        if edge not in visited_edges:
            visited_edges.append(edge)
        else:
            raise Exception(
                f"Error, edge:{edge} is a duplicate edge as it"
                + f" already is in:{visited_edges}"
            )
