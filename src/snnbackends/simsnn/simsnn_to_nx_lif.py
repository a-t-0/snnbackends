"""Conversion from simsnn (back) to nx_lif."""
import copy
from pprint import pprint
from typing import List

import networkx as nx
from simsnn.core.nodes import LIF
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snnbackends.networkx.LIF_neuron import LIF_neuron, Synapse, U, V


@typechecked
def simsnn_graph_to_nx_lif_graph(
    *,
    simsnn: Simulator,
) -> nx.DiGraph:
    """Converts sim snn graphs to nx_lif graphs.

    TODO: include timesteps.
    """
    nx_snn: nx.DiGraph = nx.DiGraph()

    # Create nx_lif neurons.
    for simsnn_lif in simsnn.network.nodes:
        lif_neuron = LIF_neuron(
            name=simsnn_lif.name,
            bias=float(simsnn_lif.bias),
            du=float(simsnn_lif.du),
            dv=float(simsnn_lif.m),
            vth=float(simsnn_lif.thr),
            pos=tuple(simsnn_lif.pos),
        )
        nx_snn.add_node(lif_neuron.full_name)
        nx_snn.nodes[lif_neuron.full_name]["nx_lif"] = [lif_neuron]

    # Create nx_lif synapses.

    for simsnn_synapse in simsnn.network.synapses:
        nx_snn.add_edges_from(
            [
                (
                    simsnn_synapse.ID[0],
                    simsnn_synapse.ID[1],
                )
            ],
            synapse=Synapse(
                weight=simsnn_synapse.w,
                delay=0,
                change_per_t=0,
            ),
        )

    # Copy graph attributes.
    nx_snn.graph = simsnn.network.graph.graph

    return nx_snn


@typechecked
def add_simsnn_simulation_data_to_reconstructed_nx_lif(
    *,
    nx_snn: nx.DiGraph,
    simsnn: Simulator,
) -> nx.DiGraph:
    """Adds the raster and multimeter data to the nx_snn that has only one
    timestep.

    Returns an nx_snn with per neuron type a list of length:
    simulation_duration x unique lif neurons, (one per timestep).
    """

    # Get the simulation duration from the multimeter.
    sim_duration: int = len(simsnn.multimeter.V)
    print(f"sim_duration={sim_duration}")

    # Verify the raster has the same simulation duration as multimeter.
    verify_nr_of_lif_neurons(nx_snn=nx_snn, expected_len=1)

    for node_index, node_name in enumerate(nx_snn.nodes):
        nx_snn.nodes[node_name]["nx_lif"].append(
            copy.deepcopy(nx_snn.nodes[node_name]["nx_lif"][0])
        )

    # Copy the lif neurons for the simulation duration.
    for t in range(1, sim_duration):
        for node_index, node_name in enumerate(nx_snn.nodes):
            # Copy spikes into nx_lif
            # TODO: verify node_index corresponds to multimeter voltage.
            # pprint(simsnn.raster.__dict__)
            nx_snn.nodes[node_name]["nx_lif"][t].spikes = bool(
                simsnn.raster.spikes[t][node_index]
            )

            # Copy voltages into nx_lif.
            # TODO: verify node_index corresponds to multimeter voltage.
            nx_snn.nodes[node_name]["nx_lif"][t].v = V(
                float(simsnn.multimeter.V[t][node_index])
            )

            # Copy amperages into nx_lif
            nx_snn.nodes[node_name]["nx_lif"][t].u = U(
                float(simsnn.multimeter.I[t][node_index])
            )

            # Create the t+1 neuron for the next timestep.
            nx_snn.nodes[node_name]["nx_lif"].append(
                copy.deepcopy(nx_snn.nodes[node_name]["nx_lif"][t])
            )

        for node_index, node_name in enumerate(nx_snn.nodes):
            # TODO: copy a_in from simsnn neuron.
            if t > 0:
                nx_snn.nodes[node_name]["nx_lif"][
                    t - 1
                ].a_in_next = get_a_in_from_sim_snn(
                    simsnn=simsnn,
                    node_name=node_name,
                    nx_snn=nx_snn,
                    t=t,
                )
                nx_snn.nodes[node_name]["nx_lif"][t].a_in = nx_snn.nodes[
                    node_name
                ]["nx_lif"][t - 1].a_in_next

        pprint(nx_snn.nodes[node_name]["nx_lif"][-1].__dict__)

        # Verify the dimensions of the outgoing nx_snn.
        # TODO: a_in or a_in_next
        verify_nr_of_lif_neurons(nx_snn=nx_snn, expected_len=t + 2)
    return nx_snn


@typechecked
def get_a_in_from_sim_snn(
    simsnn: Simulator, node_name: str, nx_snn: nx.DiGraph, t: int
) -> float:
    """Raises error if the nr of lif neurons per node is not as expected."""
    neuron: LIF
    for x in simsnn.network.nodes:
        if x.name == node_name:
            neuron = x
            break
    if not isinstance(neuron, LIF):
        raise ValueError(f"Neuron:{node_name} not found.")

    incoming_spike_neurons: List[LIF] = []
    a_in_next: float = 0
    for synapse in simsnn.network.synapses:
        # TODO: also support != 0 for .out<0.
        if (
            synapse.post == neuron
            and nx_snn.nodes[synapse.pre.name]["nx_lif"][t - 1].spikes
        ):
            print(f"{t-1} {node_name}-{synapse.pre.name}: {synapse.w}")
            incoming_spike_neurons.append(synapse.pre)
            a_in_next += synapse.w
    if len(incoming_spike_neurons) > 0:
        print(f"a_in_next={a_in_next}")
    return a_in_next


@typechecked
def verify_nr_of_lif_neurons(nx_snn: nx.DiGraph, expected_len: int) -> None:
    """Raises error if the nr of lif neurons per node is not as expected."""
    for node_name in nx_snn.nodes:
        # Verify the dimensions of the incoming nx_snn.
        node = nx_snn.nodes[node_name]
        if len(node["nx_lif"]) != expected_len:
            raise ValueError(
                f"Length lif_neurons not:{expected_len}, instead:"
                + f" it was: {len(node['nx_lif'])}"
                + f" for: {node_name}."
            )
