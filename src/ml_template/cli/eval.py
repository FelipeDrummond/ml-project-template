"""Eval entrypoint stub. Wire up per project."""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from ml_template.config_schemas import register_configs
from ml_template.utils import setup_logging

logger = logging.getLogger(__name__)

register_configs()


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    logger.info("Eval entrypoint — not implemented yet.\n%s", OmegaConf.to_yaml(cfg))
    raise NotImplementedError("Implement evaluation for your project.")


if __name__ == "__main__":
    main()
