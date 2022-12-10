"""Verifies the graph represents a connected and valid SNN, with all required
neuron and synapse properties specified."""
# Import the networkx module.

import networkx as nx
from typeguard import typechecked

from snnbackends.networkx.LIF_neuron import LIF_neuron

from .lava.verify_graph_is_lava_snn import (
    verify_lava_neuron_properties_are_specified,
)
from .networkx.verify_graph_is_networkx_snn import (
    assert_synapse_properties_are_specified,
)


@typechecked
def verify_networkx_snn_spec(
    snn_graph: nx.DiGraph, t: int, backend: str
) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.
    :param G: The original graph on which the MDSA algorithm is ran.

    """
    for nodename in snn_graph.nodes:
        # TODO: expect list of neurons, instead of single neuron.
        if backend in ["nx", "generic"]:
            print(f"nodename={nodename}")
            lif_neurons = snn_graph.nodes[nodename]["nx_lif"]
            if not isinstance(lif_neurons[t], LIF_neuron):
                raise ValueError(
                    f"Error, neuron is not of type:{LIF_neuron}, instead it"
                    + f" is of type:{type(lif_neurons[t])}"
                )
        elif backend == "lava":
            verify_lava_neuron_properties_are_specified(
                snn_graph.nodes[lif_neurons[t]]
            )
        else:
            raise ValueError(f"Backend:{backend} not supported.")
    #
    # TODO: verify synapse properties
    for edge in snn_graph.edges:
        assert_synapse_properties_are_specified(snn_graph, edge)
