"""Manages a run on simsnn."""


from typing import List

from simsnn.core.nodes import LIF
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
        for t in range(sim_duration):
            # TODO: Verify the neurons of the previous timestep are valid.
            snn.run(
                1, plotting=False, extend_multimeter=True, extend_raster=True
            )
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
    """Returns True if the MDSA snn is done, False otherwise.

    TODO: https://github.com/a-t-0/snncompare/issues/195
    """

    # Get the non RandomSpiker neurons. (Get the normal neurons).
    non_random_nodes: List[LIF] = []
    for node in snn.network.nodes:
        if isinstance(node, LIF):
            non_random_nodes.append(node)

    # Check if any of the selector or next round neurons still spike.
    if (
        not any(
            neuron_identifier in node.name and node.out > 0
            for node in non_random_nodes
            for neuron_identifier in ["selector", "next_round"]
        )
        and t > 0
    ):
        return True

    # if all_terminator_nodes_have_spiked(snn=snn):
    # return True
    return False


@typechecked
def all_terminator_nodes_have_spiked(
    snn: Simulator,
) -> bool:
    """Returns True if all terminator neurons have spiked."""
    # Count nr of terminators
    terminator_nodes: List = []
    for node in snn.network.nodes:
        if isinstance(node, LIF):
            if "terminator" in node.name:
                terminator_nodes.append(node)

    if all(node.out > 0 for node in terminator_nodes):
        return True
    return False
