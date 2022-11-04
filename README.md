# Spiking Neural Network Simulator Backends

[![Python 3.10][python_badge]](https://www.python.org/downloads/release/python-3106/)
[![License: AGPL v3][agpl3_badge]](https://www.gnu.org/licenses/agpl-3.0)
[![Code Style: Black][black_badge]](https://github.com/ambv/black)
[![Code Coverage][codecov_badge]](https://codecov.io/gh/a-t-0/snnalgos)

This is a library of Spiking Neural Network (SNN) backends. They take an networkx
graph, that specifies an SNN, as input, and then simulate the network on the
chosen backend.

## Parent Repository

These algorithms can be analysed using
[this parent repository].
Together, these repos can be used to investigate the effectivity of various
[brain-adaptation] mechanisms applied to these [algorithms], in order to increase
their [radiation] robustness. You can run it on various backends, as well as on
a custom LIF-neuron simulator.

## Algorithms

Here is a list of supported neuromorphic backends:

| Algorithm                            | Encoding | Adaptation | Radiation    |
| ------------------------------------ | -------- | ---------- | ------------ |
| Minimum Dominating Set Approximation | Sparse   | Redundancy | Neuron Death |
|                                      |          |            |              |
|                                      |          |            |              |

<!-- Un-wrapped URL's (Badges and Hyperlinks) -->

[agpl3_badge]: https://img.shields.io/badge/License-AGPL_v3-blue.svg
[algorithms]: https://github.com/a-t-0/snnalgos
[black_badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[brain-adaptation]: https://github.com/a-t-0/snnadaptation
[codecov_badge]: https://codecov.io/gh/a-t-0/snn/branch/main/graph/badge.svg
[python_badge]: https://img.shields.io/badge/python-3.10-blue.svg
[radiation]: https://github.com/a-t-0/snnradiation
[this parent repository]: https://github.com/a-t-0/snncompare
