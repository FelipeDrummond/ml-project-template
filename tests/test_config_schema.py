"""Pin the config-validation contract.

These tests exist so that if someone refactors `register_configs()` or the
top-level `Config` dataclass and silently weakens type checking on YAML /
CLI overrides, CI catches it. Without these, schema drift is invisible
until a typo slips into a real run.
"""

from __future__ import annotations

import pytest
from hydra import compose, initialize
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from ml_template.config_schemas import Config, register_configs


@pytest.fixture(autouse=True, scope="module")
def _register() -> None:
    register_configs()


def test_default_config_loads_as_typed_dataclass() -> None:
    """`OmegaConf.to_object` must produce a real `Config` instance."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    typed = OmegaConf.to_object(cfg)
    assert isinstance(typed, Config)
    assert isinstance(typed.data.n_samples, int)
    assert isinstance(typed.trainer.lr, float)


def _full_error(exc: BaseException) -> str:
    """Concatenate exception message + chained cause messages.

    Hydra wraps OmegaConf's ConfigKeyError / ValidationError in a
    ConfigCompositionException whose top-level message is generic
    ("Error merging override ..."). The actual validation reason is in the
    chained `__cause__`. This helper makes both visible to assertions.
    """
    parts = [str(exc)]
    cur = exc.__cause__
    while cur is not None:
        parts.append(str(cur))
        cur = cur.__cause__
    return " | ".join(parts)


def test_typo_in_field_name_rejected() -> None:
    """A typo on an existing field must fail at compose time.

    Note: Hydra's `+key=val` syntax explicitly *adds* a new field and
    bypasses struct mode. Plain `key=val` on an unknown field — i.e. a typo
    — is rejected, which is the case we want to lock down.
    """
    with (
        initialize(version_base=None, config_path="../configs"),
        pytest.raises(ConfigCompositionException) as excinfo,
    ):
        compose(config_name="config", overrides=["data.not_a_real_field=1"])
    assert "not_a_real_field" in _full_error(excinfo.value)


def test_wrong_type_on_int_field_rejected() -> None:
    with (
        initialize(version_base=None, config_path="../configs"),
        pytest.raises(ConfigCompositionException) as excinfo,
    ):
        compose(config_name="config", overrides=["data.n_samples=not_a_number"])
    assert "Integer" in _full_error(excinfo.value)


def test_wrong_type_on_float_field_rejected() -> None:
    with (
        initialize(version_base=None, config_path="../configs"),
        pytest.raises(ConfigCompositionException) as excinfo,
    ):
        compose(config_name="config", overrides=["trainer.lr=oops"])
    assert "Float" in _full_error(excinfo.value)


def test_example_experiment_composes() -> None:
    """The shipped example experiment must validate against the schema."""
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config", overrides=["experiment=example"])
    typed = OmegaConf.to_object(cfg)
    assert isinstance(typed, Config)
    assert typed.mlflow.experiment_name == "example"
