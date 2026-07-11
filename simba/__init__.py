# SIMBA — Simple Intelligence Module for Basic Ablations.
#
# A from-scratch, minimal re-implementation of the TIGER controller brain:
# one plain DQN decision module, one prototypical inference module, one
# transparent budget environment. No Rainbow tweaks, no Kafka, no node
# features, no health monitoring. Traffic only: flowstats + packets.
#
# Usable in two modalities, like the original TigerBrain:
#   * offline  — replaying recorder-format traces (or synthetic ones),
#                see simba_offline.py at the repo root;
#   * online   — inside the POX controller in GNS3, via
#                SimbaBrain.process_input(flows) (see tiger_server.py).

from .config import SimbaConfig  # noqa: F401
