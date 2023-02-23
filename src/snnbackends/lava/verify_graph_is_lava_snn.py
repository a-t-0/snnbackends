"""Verifies the graph represents a connected and valid SNN, with all required
neuron and synapse properties specified."""
# Import the networkx module.

import networkx as nx
import numpy as np
from typeguard import typechecked


@typechecked
def verify_lava_neuron_properties_are_specified(
    *,
    node: nx.DiGraph.nodes,
) -> None:
    """

    :param node: nx.DiGraph.nodes:
    :param node: nx.DiGraph.nodes:

    """
    bias = node["lava_LIF"].bias_mant.get()
    if not isinstance(bias, (float, np.ndarray)):
        # TODO: include additional verifications on dimensions of bias.
        raise TypeError(
            f"Bias is not a np.ndarray, it is of type:{type(bias)}."
        )

    if not isinstance(node["lava_LIF"].du.get(), float):
        raise TypeError("du is not a float.")
    if not isinstance(node["lava_LIF"].dv.get(), float):
        raise TypeError("dv is not a float.")
    if not isinstance(node["lava_LIF"].vth.get(), float):
        raise TypeError("vth is not a float.")
