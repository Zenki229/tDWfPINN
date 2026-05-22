import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf

from libs.jax_run_metadata import build_run_metadata, prepare_wandb_run


def _cfg(case="lshape", alpha=1.25, method="GJ-II"):
    return OmegaConf.create({
        "seed": 42,
        "pde": {
            "name": case,
            "al": alpha,
            "method": method,
            "k": 1,
            "lam": 1.0,
            "GJ": {"nums": 64},
            "MC": {"nums": 640, "eps": 1e-8},
        },
        "training": {"max_steps": 5000},
        "wandb": {
            "project": "tDWfPINN",
            "entity": None,
            "mode": "disabled",
            "name": None,
            "group": None,
            "tags": None,
            "job_type": None,
        },
    })


def test_build_run_metadata_includes_alpha_method_and_quadrature():
    meta = build_run_metadata(_cfg())

    assert meta["name"] == "lshape_alpha1p25_GJ-II_GJ64_steps5000_seed42"
    assert meta["group"] == "lshape_alpha1p25"
    assert {"jax", "lshape", "alpha1p25", "GJ-II", "GJ", "typeII"} <= set(meta["tags"])


def test_prepare_wandb_run_preserves_explicit_name_and_adds_tags():
    cfg = _cfg(case="forward", alpha=1.75, method="MC-I")
    cfg.wandb.name = "manual-name"
    cfg.wandb.tags = ["server"]

    prepare_wandb_run(cfg)

    assert cfg.wandb.name == "manual-name"
    assert cfg.wandb.group == "forward_alpha1p75"
    assert "server" in cfg.wandb.tags
    assert "MC-I" in cfg.wandb.tags
