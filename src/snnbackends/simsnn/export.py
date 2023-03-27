"""Converts simsnn network into exportable json object."""
from typing import Dict, List

from simsnn.core.connections import Synapse
from simsnn.core.networks import Network
from simsnn.core.nodes import LIF
from simsnn.core.simulators import Simulator
from typeguard import typechecked

from snnbackends.simsnn.helper import assert_contains_no_dupes


# pylint: disable=R0903
class Json_dict_simsnn:
    """Converts a simsnn Simulator object into an exportable json dict."""

    @typechecked
    def __init__(
        self,
        simulator: Simulator,
    ) -> None:
        self.nodes: Dict[str, Dict] = simsnn_nodes_to_json(
            nodes=simulator.network.nodes
        )
        self.synapses = simsnn_synapses_to_json(
            synapses=simulator.network.synapses
        )
        self.graph = simulator.network.graph.graph


@typechecked
def simsnn_nodes_to_json(*, nodes: List[LIF]) -> Dict[str, Dict]:
    """Converts simsnn nodes into an exportable json dict."""
    json_nodes: Dict[str, Dict] = {}
    node_names: List[str] = list(map(lambda x: x.name, nodes))
    for node in nodes:
        assert_contains_no_dupes(some_list=node_names)
        json_nodes[node.name] = node.__dict__
    return json_nodes


@typechecked
def simsnn_synapses_to_json(*, synapses: List[Synapse]) -> List[Dict]:
    """Converts simsnn synapses into an exportable json dict."""
    json_synapses: List[Dict] = []
    for synapse in synapses:
        json_synapses.append(
            {key: synapse.__dict__[key] for key in ("ID", "index", "w")}
        )
        # print(f"synapse={synapse.__dict__}")
        json_synapses[-1]["d"] = len(synapse.out_pre.tolist())
        json_synapses[-1]["out_pre"] = synapse.out_pre.tolist()

    return json_synapses


@typechecked
def json_to_simsnn(*, json_simsnn: Dict) -> Simulator:
    """Converts exportable json dict int simsnn Simulator."""
    net = Network()

    simsnn_nodes = json_to_simsnn_nodes(
        json_nodes=json_simsnn["nodes"], net=net
    )
    json_to_simsnn_synapses_in_net(
        simsnn_nodes=simsnn_nodes,
        json_synapses=json_simsnn["synapses"],
        net=net,
    )

    # Also load completed stages from graph attribute.
    net.graph.graph = json_simsnn["graph"]
    sim = Simulator(net, monitor_I=True)

    # TODO: restore check: if add_to_raster:
    # Add all neurons to the raster.
    sim.raster.addTarget(net.nodes)
    # TODO: restore check: if add_to_multimeter:
    # Add all neurons to the multimeter.
    sim.multimeter.addTarget(net.nodes)

    return sim


@typechecked
def json_to_simsnn_nodes(
    *, json_nodes: Dict[str, Dict], net: Network
) -> Dict[str, LIF]:
    """Converts exportable json dict into simsnn nodes."""
    node_names: List[str] = list(json_nodes.keys())
    assert_contains_no_dupes(some_list=node_names)

    simsnn: Dict[str, LIF] = {}
    for node_name, json_node in json_nodes.items():
        # exclude_keys = ["I", "out", "V", "spike_only_if_thr_exceeded"]
        exclude_keys = ["I", "out", "V"]
        lif_dict = {
            key: json_node[key]
            for key in set(list(json_node.keys())) - set(exclude_keys)
        }
        lif_dict["amplitude"] = 1
        simsnn[node_name] = net.createLIF(**lif_dict)

    return simsnn


@typechecked
def json_to_simsnn_synapses_in_net(
    *, net: Network, simsnn_nodes: Dict[str, LIF], json_synapses: List[Dict]
) -> None:
    """Converts exportable json dict into simsnn nodes."""
    for json_synapse in json_synapses:
        net.createSynapse(
            pre=simsnn_nodes[json_synapse["ID"][0]],
            post=simsnn_nodes[json_synapse["ID"][1]],
            ID=json_synapse["ID"],
            w=json_synapse["w"],
            d=json_synapse["d"],
        )
