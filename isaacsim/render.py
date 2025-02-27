"""Generate infrared rendering using replicator
"""
import json
import math
import os
import random
import sys

import carb
import yaml
from omni.isaac.kit import SimulationApp

from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
import hydra

# hydra: load config
with initialize(version_base=None, config_path="config", job_name="replicator_ir"):
    cfg = compose(config_name="hssd.yaml" , overrides=sys.argv[1:])

if cfg["seed"] >= 0:
    random.seed(cfg["seed"])

# start simulation
_app = SimulationApp(launch_config=cfg['launch_config'])
_Log = _app.app.print_and_log

from omni.isaac.core import World
from replicator import IRReplicator

# main program 
def run(cfg: DictConfig) -> None:
    _Log("start running")
    _world = World(set_defaults=True) #**cfg['world'],  
    _world.set_simulation_dt(**cfg["world"])
    
    # start replicator
    rep = IRReplicator(_app, _world, cfg)
    rep.start()

    _Log("keep GUI running if headless is False")
    while _app.is_running() and not cfg['launch_config']['headless']:
        _world.step(render=True)

    _app.close()

if __name__ == "__main__":
    run(cfg)
