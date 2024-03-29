"""Converts networkx graph representing lava spiking-neural-network into
SNN."""
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx

# Instantiate Lava processes to build network
import numpy as np
from lava.proc.dense.process import Dense
from lava.proc.lif.process import LIF
from networkx.classes.digraph import DiGraph
from typeguard import typechecked

from ..networkx.verify_graph_is_networkx_snn import (
    assert_synapse_properties_are_specified,
)


@typechecked
def initialise_networkx_to_snn_conversion(
    *,
    G: DiGraph,
) -> Tuple[List[int], LIF, List[LIF], int, Dict[LIF, int]]:
    """Prepares a networkx graph G to be converted into a Lava-nc neural
    network.

    :param G: The original graph on which the MDSA algorithm is ran. Networkx
    graph that specifies the Lava neural network.
    """

    # 1. Start with first incoming node.
    # pylint: disable=R0801
    # TODO: remove old_conversion and then remove the necessitiy for this
    # pylint disable command.
    first_node = list(G.nodes)[0]

    # Create dictionary with Lava LIF neurons as keys, neuron names as values.
    neuron_dict: Dict = {}

    (
        converted_nodes,
        lhs_neuron,
        neurons,
        lhs_node_name,
        neuron_dict,
        _,
    ) = convert_networkx_to_lava_snn(
        G=G,
        converted_nodes=[],
        neurons=[],
        lhs_node_name=first_node,
        visited_nodes=[],
        neuron_dict=neuron_dict,
    )
    return converted_nodes, lhs_neuron, neurons, lhs_node_name, neuron_dict


@typechecked
def convert_networkx_to_lava_snn(
    *,
    G: DiGraph,
    converted_nodes: List[int],
    neurons: List[LIF],
    lhs_node_name: int,
    visited_nodes: List[int],
    neuron_dict: Dict[LIF, int],
) -> Tuple[List[int], LIF, List[LIF], int, Dict[LIF, int], List[int]]:
    """Recursively converts the networkx graph G into a Lava SNN.

    :param G: The original graph on which the MDSA algorithm is ran. Networkx
     graph that specifies the Lava neural network.
    :param converted_nodes: List of networkx node_names that already have been
    converted to the Lava SNN.
    :param neurons: List of Lava neuron objects.
    :param lhs_node_name: The left-hand-side node_name that is taken as a
    start point per recursive evaluation. All the neighbours are the
    right-hand-side neurons.
    :param visited_nodes: Nodes that have been the lhs node in this recursive
    conversion function. Neighbours are also converted, so neighbours can be
    converted but not visited.
    :param neuron_dict: Dictionary with Lava neuron objects as keys, and the
    node_name as items. (Default value = {})
    """
    # pylint: disable=too-many-arguments
    visited_nodes.append(lhs_node_name)

    # Incoming node, if it is not yet converted, then convert to neuron.
    if not node_is_converted(
        converted_nodes=converted_nodes, node_name=lhs_node_name
    ):
        (
            converted_nodes,
            lhs_neuron,
            neurons,
            lhs_node_name,
        ) = create_neuron_from_node(
            G=G,
            converted_nodes=converted_nodes,
            neurons=neurons,
            node_name=lhs_node_name,
            t=0,
        )
    else:
        lhs_neuron = get_neuron_belonging_to_node_from_list(
            neurons=neurons, node_name=lhs_node_name, nodes=converted_nodes
        )

    # For all edges of node, if synapse does not yet exists:
    # Is a set  because bi-directional edges create neighbour duplicates.
    # pylint: disable=R0801
    # Duplicate code is temporary until old code is deleted.
    for neighbour in set(nx.all_neighbors(G, lhs_node_name)):
        if neighbour not in visited_nodes:
            # Convert the neighbour neurons of the lhs_node_name into a Lava
            # neuron.
            if not node_is_converted(
                converted_nodes=converted_nodes, node_name=neighbour
            ):
                (
                    converted_nodes,
                    rhs_neuron,
                    neurons,
                    _,
                ) = create_neuron_from_node(
                    G=G,
                    converted_nodes=converted_nodes,
                    neurons=neurons,
                    node_name=neighbour,
                    t=0,
                )
            else:
                # pylint: disable=R0801
                # Duplicate code is temporary until old code is deleted.
                # Even if the neighbour is already converted, the lhs and rhs
                # neurons are still retrieved to create a synapse between them.
                lhs_neuron = get_neuron_belonging_to_node_from_list(
                    neurons=neurons,
                    node_name=lhs_node_name,
                    nodes=converted_nodes,
                )
                rhs_neuron = get_neuron_belonging_to_node_from_list(
                    neurons=neurons, node_name=neighbour, nodes=converted_nodes
                )

            # Create neuron dictionary, LIF objects as keys, neuron
            # descriptions as values.
            neuron_dict = add_neuron_to_dict(
                neighbour=neighbour,
                neuron_dict=neuron_dict,
                rhs_neuron=rhs_neuron,
            )

            # Create synapse between lhs neuron and neighbour/rhs neuron.
            lhs_neuron = add_synapse_between_nodes(
                G=G,
                lhs_neuron=lhs_neuron,
                lhs_node_name=lhs_node_name,
                neighbour=neighbour,
                rhs_neuron=rhs_neuron,
            )

        # At the first time this function is called, initialise the dictionary.
        if len(visited_nodes) == 1:
            neuron_dict = add_neuron_to_dict(
                neighbour=lhs_node_name,
                neuron_dict=neuron_dict,
                rhs_neuron=lhs_neuron,
            )

    # Recursively call that function on the neighbour neurons until no
    # new neurons are discovered.
    for neighbour in nx.all_neighbors(G, lhs_node_name):
        if neighbour not in visited_nodes:
            # pylint: disable=R0801
            # No other way is found to retrieve the properties at this point,
            # hence the call to get the properties can be duplicated elsewhere.
            (
                converted_nodes,
                _,
                neurons,
                _,
                neuron_dict,
                visited_nodes,
            ) = convert_networkx_to_lava_snn(
                G=G,
                converted_nodes=converted_nodes,
                neurons=neurons,
                lhs_node_name=neighbour,
                visited_nodes=visited_nodes,
                neuron_dict=neuron_dict,
            )
    return (
        converted_nodes,
        lhs_neuron,
        neurons,
        lhs_node_name,
        neuron_dict,
        visited_nodes,
    )


@typechecked
def node_is_converted(*, converted_nodes: List[int], node_name: int) -> bool:
    """Verifies that the incoming node is not converted into a neuron yet.

    :param converted_nodes: List of networkx node_names that already have been
    converted to the Lava SNN.
    :param neurons: List of Lava neuron objects.
    :param node_name: Node of the name of a networkx graph. Name of the node of
     the networkx graph.
    """
    return node_name in converted_nodes


# pylint: disable=R0913
@typechecked
def create_neuron_from_node(
    *,
    G: DiGraph,
    converted_nodes: List[int],
    neurons: List[LIF],
    node_name: int,
    t: int = 0,
) -> Tuple[List[int], LIF, List[LIF], int]:
    """Creates a lava LIF neuron object based on the settings of a node
    specified in the SNN graph setting.

    # TODO: change G to indicate what the type/level of the graph is. E.g.
    # original graph
    # MDSA SNN implementation graph etc.

    :param G: The original graph on which the MDSA algorithm is ran. Networkx
     graph that specifies the Lava neural network.
    :param converted_nodes: List of networkx node_names that already have been
    converted to the Lava SNN.
    :param neurons: List of Lava neuron objects.
    :param node_name: Node of the name of a networkx graph. Name of the node of
    the networkx graph.
    :param old_code:  (Default value = False)
    """

    bias, du, dv, vth = get_neuron_properties(
        G=G, node_name=str(node_name), t=t
    )

    # https://github.com/lava-nc/lava/blob/release/v0.5.0/src/lava/proc/lif/
    # process.py
    # Source: https://github.com/lava-nc/lava/blob/ee2a6cf3bd05d51d0bb269a8801b
    # e1b7da8deedd/tests/lava/proc/dense/test_stdp_sim.py
    size = 1
    neuron = LIF(bias_mant=bias, du=du, dv=dv, vth=vth, shape=(size,))

    # Add recurrent synapse if it exists.
    add_recurrent_edge(G=G, node_name=node_name, neuron=neuron)
    # neuron = create_Synapse(neuron, -2)

    neurons.append(neuron)
    converted_nodes.append(node_name)
    return converted_nodes, neuron, neurons, node_name


@typechecked
def add_recurrent_edge(*, G: DiGraph, node_name: int, neuron: LIF) -> None:
    """Adds a recurrent edge to the node if it exists.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param node_name: Node of the name of a networkx graph.
    :param neuron: Lava neuron object.
    """
    if G.has_edge(node_name, node_name):
        # Compute synaptic weight.
        weight = G.edges[(node_name, node_name)]["weight"]
        create_Synapse(neuron=neuron, weight=weight)


@typechecked
def get_neuron_properties(
    *, G: nx.DiGraph, node_name: Union[int, str], t: int
) -> Tuple[float, float, float, float]:
    """Returns the bias,du,dv and vth of a node of the MDSA SNN graph.

    :param G: The original graph on which the MDSA algorithm is ran. Networkx
    graph that specifies the Lava neural network.
    :param node_name: Node of the name of a networkx graph. Name of the node of
     the networkx graph.
    """
    if int(node_name) in G.nodes:
        if "nx_lif" in G.nodes[int(node_name)]:
            bias = G.nodes[int(node_name)]["nx_lif"][t].bias.get()
            du = G.nodes[int(node_name)]["nx_lif"][t].du.get()
            dv = G.nodes[int(node_name)]["nx_lif"][t].dv.get()
            vth = G.nodes[int(node_name)]["nx_lif"][t].vth.get()
            return bias, du, dv, vth
        raise AttributeError(
            f"node does not have nx_LIF:{G.nodes[int(node_name)]}."
        )
    raise ValueError(f"node_name:{node_name} not in G.nodes:{G.nodes}.")


@typechecked
def create_Synapse(*, neuron: LIF, weight: float) -> LIF:
    """Creates a synapse from a neuron back into itself.

    :param neuron: Lava neuron object.
    :param weight: Synaptic weight.
    """
    dense = create_weighted_synapse(weight_value=weight)

    # Connect neuron to itself.
    neuron = connect_synapse(neuron_a=neuron, neuron_b=neuron, dense=dense)
    return neuron


@typechecked
def create_weighted_synapse(*, weight_value: float) -> Dense:
    """Creates a weighted synapse between neuron a and neuron b.

    :param w: Synaptic weight.
    """
    shape = (1, 1)
    # weights = np.random.randint(100, size=shape)
    # weights = [[w]]  # Needs to be this shape for a 1-1 neuron connection.
    # weights = (1,1)  # Needs to be this shape for a 1-1 neuron connection.
    # shape = (10, 10)
    # weights = (10,10)  # Needs to be this shape for a 1-1 neuron connection.
    # size = 4
    # weights = np.eye(size) * 1
    size = 1
    weights_init = np.eye(size) * weight_value

    weight_exp = 2
    num_weight_bits = 7
    sign_mode = 1

    dense = Dense(
        shape=shape,
        weights=weights_init,
        weight_exp=weight_exp,
        num_weight_bits=num_weight_bits,
        sign_mode=sign_mode,
    )
    return dense


@typechecked
def connect_synapse(*, neuron_a: LIF, neuron_b: LIF, dense: Dense) -> LIF:
    """Connects a synapse named dense from neuron a to neuron b.

    :param neuron_a: Lava neuron object for lhs neuron.
    :param neuron_b: Lava neuron object for rhs neuron.
    :param dense: Lava object representing synapse.
    """
    neuron_a.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(neuron_b.in_ports.a_in)
    return neuron_a


@typechecked
def get_neuron_belonging_to_node_from_list(
    *, neurons: List[LIF], node_name: int, nodes: List[int]
) -> LIF:
    """Returns the lava LIF neuron object that is represented by the node
    node_name of a certain graph.

    :param neurons: List of Lava neuron objects.
    :param node_name: Node of the name of a networkx graph. Name of the node of
    the networkx graph.
    :param nodes: List of node_names of networkx graph.
    """
    index = nodes.index(node_name)
    return neurons[index]


@typechecked
def add_synapse_between_nodes(
    *,
    G: DiGraph,
    lhs_neuron: LIF,
    lhs_node_name: int,
    neighbour: int,
    rhs_neuron: LIF,
) -> LIF:
    """Adds a synapse from left the left neuron to the right neuron and from
    the right neuron to the left neuron, if the respective edge exists between
    the two nodes in the graph G.

    # TODO: change name of G to indicate level/type of graph (e.g. original
    #  graph of MDSA SNN graph).

    :param G: The original graph on which the MDSA algorithm is ran. Networkx
     graph that specifies the Lava neural network.
    :param lhs_neuron: param lhs_node_name: The left-hand-side node_name that
    is taken as a
    start point per recursive evaluation. All the neighbours are the
    right-hand-side neurons.
    :param neighbour: Name of the rhs node of the networkx graph.
    :param rhs_neuron:
    :param lhs_node_name:
    """
    # TODO: ensure the synapses are created in both directions.
    lhs_neuron = add_synapse_left_to_right(
        G=G,
        lhs_neuron=lhs_neuron,
        lhs_node_name=lhs_node_name,
        neighbour=neighbour,
        rhs_neuron=rhs_neuron,
    )
    lhs_neuron = add_synapse_right_to_left(
        G=G,
        lhs_neuron=lhs_neuron,
        lhs_node_name=lhs_node_name,
        neighbour=neighbour,
        rhs_neuron=rhs_neuron,
    )
    return lhs_neuron


@typechecked
def add_synapse_left_to_right(
    *,
    G: DiGraph,
    lhs_neuron: LIF,
    lhs_node_name: int,
    neighbour: int,
    rhs_neuron: LIF,
) -> LIF:
    """Adds a synapse from left to right between two neurons if a directed edge
    exists from the left node to the right node.

    :param G: The original graph on which the MDSA algorithm is ran. Networkx
    graph that specifies the Lava neural network.
    :param lhs_neuron: param lhs_node_name: The left-hand-side node_name that
    is taken as a start point per recursive evaluation. All the neighbours are
    the right-hand-side neurons.mdsa
    :param neighbour: Name of the rhs node of the networkx graph.
    :param rhs_neuron: param lhs_neuron:
    :param lhs_node_name:
    """
    # 3. Get the edge between lhs and rhs nodes. They are neighbours
    # so they have an edge by definition.However it is a directed graph.
    edge = get_edge_if_exists(
        G=G, lhs_node_name=lhs_node_name, rhs_node=neighbour
    )

    if edge is not None:
        # 3. Assert the synapses are fully specified.
        assert_synapse_properties_are_specified(snn_graph=G, edge=edge)

        # 4. Create synapse between incoming node and neighbour.
        dense = create_weighted_synapse(weight_value=G.edges[edge]["weight"])

        # 5. Connect neurons using created synapse.
        # TODO: write function that checks if synapse is created or not.
        lhs_neuron = connect_synapse_left_to_right(
            lhs_neuron=lhs_neuron, rhs_neuron=rhs_neuron, dense=dense
        )
    return lhs_neuron


@typechecked
def add_synapse_right_to_left(
    *,
    G: DiGraph,
    lhs_neuron: LIF,
    lhs_node_name: int,
    neighbour: int,
    rhs_neuron: LIF,
) -> LIF:
    """

    :param G: The original graph on which the MDSA algorithm is ran. Networkx
     graph that specifies the Lava neural network.
    :param lhs_neuron: param lhs_node_name: The left-hand-side node_name that
     is taken as a start point per recursive evaluation. All the neighbours
     are the right-hand-side neurons.
    :param neighbour: Name of the rhs node of the networkx graph.
    :param rhs_neuron: param rhs_node:
    :param lhs_neuron:
    :param lhs_node_name:

    """
    # 3. Get the edge between lhs and rhs nodes. They are neighbours
    # so they have an edge by definition.However it is a directed graph.
    edge = get_edge_if_exists(
        G=G, lhs_node_name=neighbour, rhs_node=lhs_node_name
    )

    if edge is not None:
        # 3. Assert the synapses are fully specified.
        assert_synapse_properties_are_specified(snn_graph=G, edge=edge)

        # 4. Create synapse between incoming node and neighbour.
        dense = create_weighted_synapse(weight_value=G.edges[edge]["weight"])

        # 5. Connect neurons using created synapse.
        # TODO: write function that checks if synapse is created or not.
        lhs_neuron = connect_synapse_right_to_left(
            lhs_neuron=lhs_neuron, rhs_neuron=rhs_neuron, dense=dense
        )
    return lhs_neuron


@typechecked
def get_edge_if_exists(
    *, G: DiGraph, lhs_node_name: int, rhs_node: int
) -> Optional[Tuple[int, int]]:
    """Returns the edge object if the graph G has an edge between the two
    nodes. Returns None otherwise.

    :param G: The original graph on which the MDSA algorithm is ran. Networkx
    graph that specifies the Lava neural network.
    :param lhs_node_name: The left-hand-side node_name that is taken as a
    start point per recursive evaluation. All the neighbours are the
    right-hand-side neurons.
    :param rhs_node:
    """
    if G.has_edge(lhs_node_name, rhs_node):
        for edge in G.edges:
            if edge == (lhs_node_name, rhs_node):
                # print_edge_properties(G, edge)
                return edge
        # Verify at least an edge the other way round exists.
        if not G.has_edge(rhs_node, lhs_node_name):
            raise ValueError(
                "Would expect an edge between a node and"
                + " its neighbour in the other direction."
            )
    # Verify at least an edge the other way round exists.
    if not G.has_edge(rhs_node, lhs_node_name):
        raise ValueError(
            "Would expect an edge between a node and"
            + " its neighbour in the other direction."
        )
    return None


@typechecked
def connect_synapse_left_to_right(
    *, lhs_neuron: LIF, rhs_neuron: LIF, dense: Dense
) -> LIF:
    """Connects a synapse named dense from lhs_neuron to rhs_neuron.

    :param lhs_neuron: param rhs_neuron:
    :param dense: param rhs_neuron:
    :param rhs_neuron:
    """
    lhs_neuron.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(rhs_neuron.in_ports.a_in)
    return lhs_neuron


@typechecked
def connect_synapse_right_to_left(
    *, lhs_neuron: LIF, rhs_neuron: LIF, dense: Dense
) -> LIF:
    """Connects a synapse named dense from lhs_neuron to rhs_neuron.

    :param lhs_neuron: param rhs_neuron:
    :param dense: param rhs_neuron:
    :param rhs_neuron:
    """
    rhs_neuron.out_ports.s_out.connect(dense.in_ports.s_in)
    dense.out_ports.a_out.connect(lhs_neuron.in_ports.a_in)
    return lhs_neuron


@typechecked
def add_neuron_to_dict(
    *, neighbour: int, neuron_dict: Dict[LIF, int], rhs_neuron: LIF
) -> Dict[LIF, int]:
    """

    :param neighbour: Name of the rhs node of the networkx graph.
    :param neuron_dict: Dictionary with Lava neuron objects as keys, and the
    node_name as items.
    :param rhs_neuron:

    """
    neuron_dict[rhs_neuron] = neighbour
    return neuron_dict
