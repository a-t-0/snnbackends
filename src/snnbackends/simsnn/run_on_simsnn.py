"""Manages a run on simsnn."""

from typing import List

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

    if len(snn.raster.targets) < 1:
        raise ValueError(
            "Error, the raster did not have any neurons that are monitored."
        )

    if list(run_config.algorithm.keys()) == ["MDSA"]:
        print(f"sim_duration={sim_duration}")

        for t in range(sim_duration):
            # TODO: Verify the neurons of the previous timestep are valid.
            snn.run(
                1, plotting=False, extend_multimeter=True, extend_raster=True
            )
            print(f"t={t}\n")
            if (
                mdsa_is_done_on_simsnn(
                    snn=snn,
                    t=t,
                )
                or t == sim_duration - 1
            ):
                snn.network.graph.graph["sim_duration"] = t + 1
                snn.network.graph.graph[
                    "actual_duration"
                ] = snn.network.graph.graph["sim_duration"]
                break

        # TODO: verify snn specification.
        if snn.network.graph.graph["actual_duration"] < 2:
            raise SystemError(
                "Error, was unable to determine why algo did not complete."
                + f"sim_duration={sim_duration}"
                + "actual_duration="
                + f'{snn.network.graph.graph["actual_duration"]}'
            )

    else:
        raise NotImplementedError(
            f"Error, the {run_config.algorithm} is not yet supported."
        )


@typechecked
def mdsa_is_done_on_simsnn(
    snn: Simulator,
    t: int,
) -> bool:
    """Returns True if the MDSA snn is done, False otherwise."""

    # If the terminator node spikes, the network is done.
    for node in snn.network.nodes:
        if "terminator" in node.name and node.out > 0:
            return True

    # If the selector or next_round neurons still spike the snn is not done.
    if (
        not any(
            neuron_identifier in node.name and node.out > 0
            for node in snn.network.nodes
            for neuron_identifier in ["selector", "next_round"]
        )
        and t > 0
    ):
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
