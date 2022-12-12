"""Runs a converted networkx graph without the Lava platform.

First verifies the graph represents a connected and valid SNN, with all
required neuron and synapse properties specified. Then loops through the
network to simulate it, one neuron at a time.
"""

import copy
from typing import Dict, List

import networkx as nx
from snnalgorithms.sparse.MDSA.is_done import mdsa_is_done
from typeguard import typechecked

from ..verify_graph_is_snn import verify_networkx_snn_spec
from .LIF_neuron import LIF_neuron

# TODO: remove if not needed.
# Initialise the nodes at time t=0 (with a_in=0).
# initialise_a_in_is_zero_at_t_is_1(snn_graph, t)


@typechecked
def run_snn_on_networkx(
    run_config: Dict, snn_graph: nx.DiGraph, sim_duration: int
) -> None:
    """Runs the simulation for t timesteps using networkx, not lava.

    :param G: The original graph on which the MDSA algorithm is ran.
    :param t: int:
    """
    print(f"sim_duration={sim_duration}")
    actual_duration: int = -1
    for t in range(sim_duration):
        # Verify the neurons of the previous timestep are valid.
        verify_networkx_snn_spec(snn_graph, t, backend="nx")

        # Copy the neurons into the new timestep.
        create_neuron_for_next_timestep(snn_graph, t)

        verify_networkx_snn_spec(snn_graph, t + 1, backend="nx")
        run_simulation_with_networkx_for_1_timestep(snn_graph, t + 1)
        if mdsa_is_done(run_config, snn_graph, t):
            actual_duration = t + 1
            snn_graph.graph["sim_duration"] = actual_duration
            break

    # TODO: delete
    if actual_duration < 0:
        actual_duration = sim_duration
        # raise Exception(
        # "Error, was unable to determine why algo did not complete.")
    # Verify the network dimensions. (Ensure sufficient nodes are added.)
    verify_networkx_graph_dimensions(snn_graph, actual_duration)


@typechecked
def create_neuron_for_next_timestep(snn_graph: nx.DiGraph, t: int) -> None:
    """Creates a new neuron for the next timestep, by copying the old neuron.

    TODO: determine what to do with the synapses.
    """
    # Horrible hack add the last tuple element to the tuple.
    # List is not used because it is not hashable, which is required for the
    # nx.DiGraph nodes.

    for node_names in snn_graph.nodes:
        # Add a new element to neuron list.
        snn_graph.nodes[node_names]["nx_lif"].append(
            copy.deepcopy(snn_graph.nodes[node_names]["nx_lif"][-1])
        )
    for node_names in snn_graph.nodes:
        if len(snn_graph.nodes[node_names]["nx_lif"]) < t + 1:
            raise Exception("Error, tuple not correctly updated.")


@typechecked
def verify_networkx_graph_dimensions(
    snn_graph: nx.DiGraph, sim_duration: int
) -> None:
    """Ensures the graph contains at least sim_duration SNN neurons of a single
    name. This is because each neuron, with a single name, needs to be
    simulated for sim_duration timesteps. This simulation is done by storing
    copies of the SNN neuron in a list, one for each simulated timestep.

    The graph is expected to adhere to the following structure:
    snn_graph.nodes[nodename] stores a single node, representing a
    neuron over time. Then that node should contain a list of nx_LIF
    neurons over time in: snn_graph.nodes[node]["nx_lif"] which is of
    type: List. Then each element in that list must be of type: nx_LIF()
    neuron.
    """
    for nodename in snn_graph.nodes:
        # Assert node has nx_LIF neuron object of type list.
        if not isinstance(snn_graph.nodes[nodename]["nx_lif"], List):
            raise Exception(
                f"Error, {nodename} nx_LIF is not of type list. Instead, it is"
                f' of type:{type(snn_graph.nodes[nodename]["nx_lif"])}'
            )

        # TODO: remove the artifact last neuron, (remove the +1), it is not
        # needed.
        if not len(snn_graph.nodes[nodename]["nx_lif"]) == sim_duration + 1:
            raise Exception(
                f"Error, neuron:{nodename} did not have len:"
                + f"{sim_duration+1}. Instead, it had len:"
                + f'{len(snn_graph.nodes[nodename]["nx_lif"])}'
            )

        for t, neuron_at_time_t in enumerate(
            snn_graph.nodes[nodename]["nx_lif"]
        ):
            if not isinstance(neuron_at_time_t, LIF_neuron):
                raise Exception(
                    f"Error, {nodename} does not have a neuron of"
                    + f"type:{LIF_neuron} at t={t}. Instead, it is of type:"
                    + f"{type(neuron_at_time_t)}"
                )


@typechecked
def run_simulation_with_networkx_for_1_timestep(
    snn_graph: nx.DiGraph, t: int
) -> None:
    """Runs the networkx simulation of the network for 1 timestep. The results
    of the simulation are stored in the snn_graph.nodes network.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    # Visited edges
    visited_edges = []

    # First reset all a_in_next values for a new round of simulation.
    reset_a_in_next_for_all_neurons(snn_graph, t)

    # Compute for each node whether it spikes based on a_in, starting at t=1.
    for node_name in snn_graph.nodes:
        nx_lif = snn_graph.nodes[node_name]["nx_lif"][t]
        spikes = nx_lif.simulate_neuron_one_timestep(nx_lif.a_in)
        if spikes:

            # Propagate the output spike to the connected receiving neurons.
            for neighbour in nx.all_neighbors(snn_graph, node_name):
                if (node_name, neighbour) not in visited_edges:
                    visited_edges.append((node_name, neighbour))

                    # Check if the outgoing edge is exists and is directed.
                    if snn_graph.has_edge(node_name, neighbour):

                        # Compute synaptic weight.
                        synapse_weight = snn_graph.edges[
                            (node_name, neighbour)
                        ]["synapse"].weight

                        # TODO: Include delay.

                        # Add input signal to connected receiving neuron.
                        snn_graph.nodes[neighbour]["nx_lif"][t].a_in_next += (
                            1 * synapse_weight
                        )

                        # Update synaptic weight based on delta t of synapse.
                        synapse_weight = +snn_graph.edges[
                            (node_name, neighbour)
                        ]["synapse"].change_per_t

    # After all inputs have been computed, store a_in_next values for next
    # round into a_in of the current round to prepare for the nextsimulation
    # step.
    for node_name in snn_graph.nodes:
        nx_lif = snn_graph.nodes[node_name]["nx_lif"][t]
        nx_lif.a_in = nx_lif.a_in_next


@typechecked
def reset_a_in_next_for_all_neurons(snn_graph: nx.DiGraph, t: int) -> None:
    """Resets the a_in_next for all neurons to 0.

    :param G: The original graph on which the MDSA algorithm is ran.
    """
    for node_names in snn_graph.nodes:
        snn_graph.nodes[node_names]["nx_lif"][t].a_in_next = 0


@typechecked
def initialise_a_in_is_zero_at_t_is_1(snn_graph: nx.DiGraph, t: int) -> None:
    """

    :param G: The original graph on which the MDSA algorithm is ran.

    """
    for node in snn_graph.nodes:
        snn_graph.nodes[node]["nx_lif"][t].a_in = 0
        # snn_graph.nodes[node]["a_in"] = 0
