import csv

from omegaconf import OmegaConf

from src.train import Trainer
from src.utils.experiments import setup_wandb


def test_hybrid_loss_event_wandb_payload(monkeypatch):
    calls = []
    monkeypatch.setattr("src.train.wandb.log", lambda payload, *args, **kwargs: calls.append((payload, kwargs)))

    trainer = Trainer.__new__(Trainer)
    trainer.loss_event = 0

    Trainer._log_loss_event(trainer, "adam", 1.25, global_step=10, epoch=1)
    Trainer._log_loss_event(trainer, "lbfgs", 0.75, global_step=10, epoch=1)

    assert calls[0][0]["train/loss_event"] == 1
    assert calls[0][0]["train/loss_continuous"] == 1.25
    assert calls[0][0]["train/adam_loss"] == 1.25
    assert calls[0][0]["train/phase"] == "adam"
    assert calls[1][0]["train/loss_event"] == 2
    assert calls[1][0]["train/loss_continuous"] == 0.75
    assert calls[1][0]["train/lbfgs_loss"] == 0.75
    assert calls[1][0]["train/phase"] == "lbfgs"


def test_timing_record_writes_adam_and_lbfgs_losses(tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.wandb.log", lambda *args, **kwargs: None)

    trainer = Trainer.__new__(Trainer)
    trainer.timing_epochs = []
    trainer.timing_epoch_steps = 2
    trainer.timing_path = str(tmp_path / "timing.csv")

    with open(trainer.timing_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "step",
                "epoch_steps",
                "elapsed_seconds",
                "total_seconds",
                "average_epoch_seconds",
                "loss",
                "adam_loss",
                "lbfgs_loss",
            ],
        )
        writer.writeheader()

    Trainer._write_timing_record(
        trainer,
        step=2,
        elapsed=3.0,
        total=3.0,
        loss_value=0.5,
        adam_loss_value=0.8,
        lbfgs_loss_value=0.5,
    )

    rows = list(csv.DictReader(open(trainer.timing_path, newline="")))
    assert rows[0]["step"] == "2"
    assert rows[0]["epoch_steps"] == "2"
    assert rows[0]["loss"] == "5.00000000e-01"
    assert rows[0]["adam_loss"] == "8.00000000e-01"
    assert rows[0]["lbfgs_loss"] == "5.00000000e-01"


def test_setup_wandb_defines_hybrid_metrics(tmp_path, monkeypatch):
    defined = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.utils.experiments.wandb.init", lambda **kwargs: object())
    monkeypatch.setattr(
        "src.utils.experiments.wandb.define_metric",
        lambda *args, **kwargs: defined.append((args, kwargs)),
    )
    cfg = OmegaConf.create({
        "wandb": {
            "project": "unit",
            "entity": None,
            "group": None,
            "mode": "disabled",
        },
        "value": 1,
    })

    setup_wandb(cfg)

    assert (("train/loss_event",), {}) in defined
    assert (
        ("train/loss_continuous",),
        {"step_metric": "train/loss_event"},
    ) in defined
    assert (
        ("timing/*",),
        {"step_metric": "train/adam_step"},
    ) in defined
