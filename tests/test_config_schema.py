"""Pin the config-validation contract.

These tests exist so that if someone refactors `register_configs()` or the
top-level `Config` dataclass and silently weakens type checking on YAML /
CLI overrides, CI catches it. Without these, schema drift is invisible
until a typo slips into a real run.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf

from ml_template.config_schemas import (
    Config,
    DataConfig,
    MLflowConfig,
    ModelConfig,
    TrainerConfig,
    register_configs,
)


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


# ----------------------------------------------------------------------------
# Merge precision: an experiment override changes only what it touches.
# ----------------------------------------------------------------------------


def test_experiment_overrides_only_touched_fields() -> None:
    """Pinning the merge contract: fields not mentioned in the experiment
    file must keep their values from the relevant base group YAML.

    This caught a real `_self_`-ordering bug once; the test prevents
    regressions where someone reorders defaults or adds an interpolation
    that silently re-overrides values.
    """
    # Fields that `configs/experiment/example.yaml` explicitly touches.
    # If you add a new key to that file, update this set too.
    overridden: dict[str, set[str]] = {
        "data": {"n_samples", "batch_size"},
        "model": {"hidden_dim", "n_layers"},
        "trainer": {"epochs"},
        "mlflow": {"experiment_name", "run_name"},
    }

    with initialize(version_base=None, config_path="../configs"):
        base = OmegaConf.to_object(compose(config_name="config"))
        with_exp = OmegaConf.to_object(
            compose(config_name="config", overrides=["experiment=example"])
        )
    assert isinstance(base, Config)
    assert isinstance(with_exp, Config)

    for group_name, touched in overridden.items():
        base_group = getattr(base, group_name)
        exp_group = getattr(with_exp, group_name)
        for field_obj in dataclasses.fields(base_group):
            base_val = getattr(base_group, field_obj.name)
            exp_val = getattr(exp_group, field_obj.name)
            if field_obj.name in touched:
                assert base_val != exp_val, (
                    f"{group_name}.{field_obj.name} listed as overridden but "
                    f"base and experiment both equal {base_val!r}"
                )
            else:
                assert base_val == exp_val, (
                    f"{group_name}.{field_obj.name} unexpectedly changed: "
                    f"{base_val!r} → {exp_val!r}"
                )


# ----------------------------------------------------------------------------
# Schema parity: every key in every YAML must exist on the schema.
#
# The compose-time check only catches typos in YAMLs that we actually
# compose in tests. This static walk catches typos in *every* YAML on
# every CI run, including unused experiment files.
# ----------------------------------------------------------------------------

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# YAML keys Hydra treats as meta — not config fields.
HYDRA_META_KEYS: frozenset[str] = frozenset({"defaults", "hydra", "_self_", "_target_"})

# Top-level config fields → their schema dataclass (for nested validation).
NESTED_SCHEMAS: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "trainer": TrainerConfig,
    "mlflow": MLflowConfig,
}

# `configs/<group>/*.yaml` → schema for that group.
GROUP_SCHEMAS: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "trainer": TrainerConfig,
}


def _field_names(dc: type) -> set[str]:
    return {f.name for f in dataclasses.fields(dc)}


def _unknown_keys(data: object, schema: type, path: str) -> list[str]:
    """Return dotted paths of keys in `data` that are not fields of `schema`."""
    if not isinstance(data, dict):
        return []
    allowed = _field_names(schema)
    bad: list[str] = []
    for key, value in data.items():
        if key in HYDRA_META_KEYS:
            continue
        if key not in allowed:
            bad.append(f"{path}.{key}")
            continue
        # Recurse into nested dataclass fields if the schema declares one.
        nested_schema = NESTED_SCHEMAS.get(key)
        if nested_schema is not None:
            bad.extend(_unknown_keys(value, nested_schema, f"{path}.{key}"))
    return bad


def _yaml_files(*subdirs: str) -> list[Path]:
    out: list[Path] = []
    for sub in subdirs:
        out.extend((CONFIGS_DIR / sub).glob("*.yaml"))
    return sorted(out)


@pytest.mark.parametrize(
    ("group", "yaml_path"),
    [
        (group, p)
        for group in GROUP_SCHEMAS
        for p in (CONFIGS_DIR / group).glob("*.yaml")
    ],
)
def test_group_yaml_keys_match_schema(group: str, yaml_path: Path) -> None:
    """No `configs/<group>/*.yaml` may introduce a key absent from its schema."""
    data = yaml.safe_load(yaml_path.read_text()) or {}
    bad = _unknown_keys(data, GROUP_SCHEMAS[group], yaml_path.name)
    assert not bad, (
        f"{yaml_path} introduces keys absent from {GROUP_SCHEMAS[group].__name__}: {bad}"
    )


def test_top_level_config_keys_match_schema() -> None:
    """`configs/config.yaml` keys must be on `Config` (or be Hydra meta keys)."""
    data = yaml.safe_load((CONFIGS_DIR / "config.yaml").read_text()) or {}
    bad = _unknown_keys(data, Config, "config.yaml")
    assert not bad, f"config.yaml introduces unknown keys: {bad}"


@pytest.mark.parametrize(
    "yaml_path", list((CONFIGS_DIR / "experiment").glob("*.yaml"))
)
def test_experiment_yaml_keys_match_schema(yaml_path: Path) -> None:
    """Experiment files use `# @package _global_`, so their top-level keys
    must be on `Config` and nested subtrees on the corresponding schema."""
    data = yaml.safe_load(yaml_path.read_text()) or {}
    bad = _unknown_keys(data, Config, yaml_path.name)
    assert not bad, (
        f"{yaml_path} introduces keys absent from Config / nested schemas: {bad}"
    )
