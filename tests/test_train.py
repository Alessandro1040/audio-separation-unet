"""The training schedule - specifically the `--resume` trap.

`lr_at` is a pure function, so it is the cheapest place to pin down the behaviour that
used to make "just train longer" useless: a resumed run inherited the learning rate of
the *old* schedule, which was already at its floor, so the extra steps learned nothing.
"""
from __future__ import annotations

import pytest

from src.config import Config
from src.train import lr_at


def _cfg(lr: float = 1e-3, warmup: int = 100) -> Config:
    cfg = Config()
    cfg.train.lr = lr
    cfg.train.warmup_steps = warmup
    return cfg


def test_lr_warms_up_then_anneals_to_five_percent() -> None:
    cfg = _cfg()
    assert lr_at(0, 1000, cfg) < lr_at(50, 1000, cfg) < lr_at(99, 1000, cfg)
    assert lr_at(99, 1000, cfg) == pytest.approx(cfg.train.lr)
    assert lr_at(100, 1000, cfg) == pytest.approx(cfg.train.lr)        # peak reached
    assert lr_at(1000, 1000, cfg) == pytest.approx(0.05 * cfg.train.lr)  # floor at the end
    # and it decreases monotonically after the warm-up
    values = [lr_at(s, 1000, cfg) for s in range(100, 1001, 50)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_resume_does_not_inherit_the_old_floor() -> None:
    cfg = _cfg()
    # resuming at step 900 of a 1000-step schedule: without the offset the run would
    # spend its 100 remaining steps at ~6e-5, i.e. learn nothing
    assert lr_at(950, 1000, cfg) == pytest.approx(0.06 * cfg.train.lr, rel=0.2)
    assert lr_at(950, 1000, cfg, start_step=900) > 5 * lr_at(950, 1000, cfg)
    # the resumed schedule starts from a re-warm-up and still anneals to the floor
    assert lr_at(900, 1000, cfg, start_step=900) < lr_at(950, 1000, cfg, start_step=900)
    assert lr_at(1000, 1000, cfg, start_step=900) == pytest.approx(0.05 * cfg.train.lr)
