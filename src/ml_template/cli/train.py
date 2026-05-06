"""Hydra-driven training entrypoint.

Run with `python -m ml_template.cli.train` or `make train`.
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from ml_template.config_schemas import Config, register_configs
from ml_template.training import train as run_training
from ml_template.utils import setup_logging

logger = logging.getLogger(__name__)

register_configs()


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    logger.info("Loaded config:\n%s", OmegaConf.to_yaml(cfg))
    typed: Config = OmegaConf.to_object(cfg)  # type: ignore[assignment]
    run_training(typed)


if __name__ == "__main__":
    main()
