"""Adds a monitor dict to a lava SNN."""
from typeguard import typechecked
from lava.proc.monitor.process import Monitor

@typechecked
def add_monitor_to_dict(
    neuron: str, monitor_dict: dict, sim_time: int
) -> Monitor:
    """Creates a dictionary monitors that monitor the outgoing spikes of LIF
    neurons.

    :param neuron: Lava neuron object.
    :param monitor_dict: Dictionary of neurons whose spikes are monitored.
    :param sim_time: Nr. of timesteps for which the experiment is ran.
    :param monitor_dict: Dictionary of neurons whose spikes are monitored.
    """
    # TODO: make this typing sensible.
    if not isinstance(neuron, str):
        monitor = Monitor()
        monitor.probe(neuron.out_ports.s_out, sim_time)
        monitor_dict[neuron] = monitor
    else:
        raise Exception(
            "Error, neuron object type is not str, it " + f"is:{type(neuron)}"
        )
    return monitor_dict
