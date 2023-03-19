"""Manages a run on simsnn."""

from typing import List, Tuple

from simsnn.core.simulators import Simulator
from snncompare.run_config.Run_config import Run_config
from typeguard import typechecked


@typechecked
def run_snn_on_simsnn(
    *,
    run_config: Run_config,
    snn: Simulator,
    sim_duration: int,
) -> None:
    """Runs the simulation for t timesteps using simsnn."""
    actual_duration: int = -1

    if len(snn.raster.targets) < 1:
        raise ValueError(
            "Error, the raster did not have any neurons that are monitored."
        )

    # TODO: verify snn specification.
    if list(run_config.algorithm.keys()) == ["MDSA"]:
        (
            still_running_indices,
            terminator_indices,
        ) = get_mdsa_completion_neuron_indices(snn=snn)

        for t in range(sim_duration):
            # TODO: Verify the neurons of the previous timestep are valid.

            snn.run(1, plotting=False)

            if mdsa_is_done_on_simsnn(
                snn=snn,
                still_running_indices=still_running_indices,
                terminator_indices=terminator_indices,
            ):
                actual_duration = t + 1
                snn.network.graph.graph["sim_duration"] = actual_duration
                snn.network.graph.graph["actual_duration"] = actual_duration
                break
            # pprint(snn.raster.spikes)

    else:
        raise NotImplementedError(
            f"Error, the {run_config.algorithm} is not yet supported."
        )

    # TODO: delete
    if actual_duration < 0:
        actual_duration = sim_duration


@typechecked
def mdsa_is_done_on_simsnn(
    snn: Simulator,
    still_running_indices: List[int],
    terminator_indices: List[int],
) -> bool:
    """Returns True if the MDSA snn is done, False otherwise."""
    for terminated_index in terminator_indices:
        if snn.raster.spikes[0][terminated_index]:
            print("DONE WITH SIMULATION")
            return True
    if not any(
        snn.raster.spikes[0][still_running_index]
        for still_running_index in still_running_indices
    ):
        print("NOT STILL RUNNING.")
        return True
    return False


@typechecked
def get_mdsa_neuron_indices_for_spike_raster(
    *,
    node_name_identifier: str,
    snn: Simulator,
) -> List[int]:
    """Returns a list of integers to indicate the positions in which neurons of
    a certain type are located in the spike raster."""
    indices: List[int] = []
    node_names: List[str] = list(map(lambda x: x.name, snn.network.nodes))
    for index, node_name in enumerate(node_names):
        if node_name_identifier in node_name:
            indices.append(index)
    return indices


@typechecked
def get_mdsa_completion_neuron_indices(
    *,
    snn: Simulator,
) -> Tuple[List[int], List[int]]:
    """Returns a boolean for each neuron type that is used to determine whether
    the MDSA snn is done. These neurons are:

    - terminator_node: If any of these spike, the network is done.

    Radiation may prevent terminator nodes from working, so check if anything
    still spikes:
    - selector_node: if any of these spike, the network is still running.
    - next_round_node: if any of these spike, the network is still running.
    Since the selector_node and next_round_node are used in the same way, their
    indices are concatenated for a single check.
    """
    # Get indices for terminator_node.
    terminator_indices: List[int] = get_mdsa_neuron_indices_for_spike_raster(
        node_name_identifier="terminator",
        snn=snn,
    )
    still_running_indices: List[
        int
    ] = get_mdsa_neuron_indices_for_spike_raster(
        node_name_identifier="selector",
        snn=snn,
    )
    still_running_indices.extend(
        get_mdsa_neuron_indices_for_spike_raster(
            node_name_identifier="next_round",
            snn=snn,
        )
    )

    return terminator_indices, still_running_indices
