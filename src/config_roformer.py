"""Configuration for the Mel-band RoFormer path (the U-Net one lives in `src/config.py`).

Everything that is shared with the U-Net - `STFTConfig`, `DataConfig`, `TrainConfig` - is
imported from `src/config.py` on purpose, so the two architectures are trained by the same
data pipeline, the same schedule and the same EMA code, and their numbers are comparable.
Only what is genuinely new is defined here:

* `RoFormerModelConfig` - the band/attention hyper-parameters;
* `RoFormerLossConfig` - the multi-resolution STFT loss of the RoFormer recipe.

Nothing in `src/config.py` is touched, so existing checkpoints keep loading.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .config import DataConfig, STFTConfig, TrainConfig


@dataclass
class RoFormerModelConfig:
    """Band-split RoFormer hyper-parameters (see `src/models/roformer.py`)."""

    dim: int = 128              # transformer width; 192-256 is the paper's range
    depth: int = 4              # blocks, each with band attention + time attention
    heads: int = 4
    ff_mult: int = 4
    n_bands: int = 64           # *target* mel bands; wide ones are split (see max_band_bins)
    band_overlap: float = 0.25  # how much neighbouring bands share bins
    fmin: float = 30.0
    fmax: float | None = None   # None = up to Nyquist (nothing is left without a mask)
    max_band_bins: int = 32     # bands are padded to the widest: this bounds the memory
    n_fft_extra: int = 1024     # second STFT resolution (0 disables the multi-res input)
    mask: str = "tanh"          # tanh -> complex ratio mask, sigmoid -> real gain
    per_channel_mask: bool = False   # False shares one mask between L and R (cheaper)
    dropout: float = 0.0


@dataclass
class RoFormerLossConfig:
    """Waveform L1 + multi-resolution STFT loss (the recipe the paper family uses).

    `magnitude`/`magnitude_compress` reproduce the U-Net's single-resolution compressed
    magnitude term; they are off by default because the multi-resolution term below is
    strictly more informative, but an A/B against `src/train.py` should set
    `loss.magnitude: 0.5`, `loss.mrstft: 0.0` so that *only* the architecture differs.
    """

    waveform: float = 1.0
    magnitude: float = 0.0
    magnitude_compress: float = 0.3
    mrstft: float = 1.0
    mrstft_ffts: list[int] = field(default_factory=lambda: [2048, 1024, 512, 256])
    mrstft_log_weight: float = 1.0    # weight of the log-magnitude term of each resolution
    si_sdr: float = 0.0               # -SI-SDR/10 term, off by default
    consistency: bool = True


@dataclass
class RoFormerConfig:
    stft: STFTConfig = field(default_factory=STFTConfig)
    model: RoFormerModelConfig = field(default_factory=RoFormerModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: RoFormerLossConfig = field(default_factory=RoFormerLossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_SECTIONS: dict[str, type] = {
    "stft": STFTConfig,
    "model": RoFormerModelConfig,
    "data": DataConfig,
    "loss": RoFormerLossConfig,
    "train": TrainConfig,
}


def _dataclass_from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Instantiate `cls` from a dict, ignoring unknown keys (forward compatible)."""
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})  # type: ignore[call-arg]


def _from_raw(raw: dict[str, Any]) -> RoFormerConfig:
    cfg = RoFormerConfig()
    for section, cls in _SECTIONS.items():
        values = dict(raw.get(section) or {})
        if section == "data" and "aug" in values:
            from .config import AugmentConfig

            cfg.data.aug = _dataclass_from_dict(AugmentConfig, dict(values.pop("aug") or {}))
        setattr(cfg, section, _dataclass_from_dict(cls, values))
    return cfg


def config_from_payload(payload: dict[str, Any]) -> RoFormerConfig:
    """Rebuild the config stored inside a RoFormer checkpoint."""
    if not payload:
        raise KeyError("checkpoint has no stored config")
    return _from_raw(payload)


def load_roformer_config(path: str | Path | None = None, **overrides: Any) -> RoFormerConfig:
    """Load a YAML config (optional) and apply dotted `section.key` overrides (`--set`)."""
    raw: dict[str, Any] = {}
    if path is not None:
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = _from_raw(raw)
    for key, value in overrides.items():
        if value is None:
            continue
        section, _, name = key.partition(".")
        if not name:
            raise KeyError(f"override '{key}' must look like 'train.lr'")
        target = getattr(cfg, section)
        if not hasattr(target, name):
            raise KeyError(f"unknown config field '{section}.{name}'")
        setattr(target, name, value)
    return cfg


def save_roformer_config(cfg: RoFormerConfig, path: str | Path) -> None:
    import yaml

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
