"""Mel-band RoFormer: the band-split + rotary-attention family, written from scratch.

This is the second architecture in the repository. It is *not* a U-Net: instead of
convolving over a spectrogram image, it cuts the frequency axis into mel-spaced bands
(each band is a token), projects every band with its own linear layer, and then runs
transformer blocks that attend

  * **across bands**, with rotary position embeddings on the band index - this is how a
    source can be tracked by its harmonic series (partial 1 of the bass and partial 8 of
    the same bass are far apart in frequency, and a learned band embedding cannot say
    "these belong together", a rotation can);
  * **across time**, with rotary embeddings on the frame index - this is how a source is
    tracked through a note whose pitch glides.

The design follows the family that currently holds the state of the art in music source
separation, where a plain U-Net is no longer competitive:

  * Wang et al., *Music Source Separation with Band-split RoPE Transformer* (BS-RoFormer),
    2023/2024 - band split + RoPE attention;
  * Lu et al., *Mel-Band RoFormer for Music Source Separation*, 2023 - the mel-spaced band
    split used here, which beats the plain band split because the bands are wide where
    hearing is coarse and narrow where it is fine;
  * the multi-resolution input is the other MDX23-winning trick: the model also sees a
    second, shorter STFT of the mixture, so attacks (high time resolution) and low
    pitched content (high frequency resolution) reach the band tokens at once.

Reference implementation to compare against: `lucidrains/BS-RoFormer`. This one is
deliberately small enough to train on a laptop M5, and it keeps the repository's existing
contract: it emits the same complex masks (`tanh`, cIRM) as `src/models/unet.py`, and its
wrapper (`src/models/roformer_separation.py`) is drop-in compatible with
`SpectrogramSeparator` - `separate_long`, the loss and the trainer all work unchanged.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def hz_to_mel(freq: float) -> float:
    """Hz -> mel with the (HTK/O'Neil) formula used by the mel-band papers."""
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def band_hz_ranges(
    n_bands: int,
    fmin: float,
    fmax: float | None,
    band_overlap: float,
    sample_rate: int,
    n_fft: int,
    max_band_bins: int,
) -> list[tuple[float, float]]:
    """Mel-spaced, overlapping (lo_hz, hi_hz) bands, in *Hertz* on purpose.

    Working in Hz here is what lets a second STFT resolution (see `n_fft_extra`) use the
    very same bands: the band list is computed once for the primary `n_fft` and then only
    re-quantised to the other FFT grid, so the two feature sets line up band by band.

    The bands always reach up to Nyquist: audio above the mel range (`fmax`, default: all
    the way up) is covered by extra bands of the same maximum width, so *every* bin is
    predicted by some band instead of being silently dropped for every source at once.
    """
    nyquist = sample_rate / 2.0
    fmax = nyquist if fmax is None else min(float(fmax), nyquist)
    lo_mel, hi_mel = hz_to_mel(fmin), hz_to_mel(fmax)
    bands: list[tuple[float, float]] = []
    for i in range(n_bands):
        lo = mel_to_hz(lo_mel + (hi_mel - lo_mel) * i / n_bands)
        hi = mel_to_hz(lo_mel + (hi_mel - lo_mel) * (i + 1) / n_bands)
        pad = (hi - lo) * band_overlap / 2.0
        bands.append((max(lo - pad, 0.0), hi + pad))
    if fmax < nyquist or bands[-1][1] < nyquist:
        bands[-1] = (bands[-1][0], nyquist)     # the remainder of the spectrum

    def width_in_bins(band: tuple[float, float]) -> int:
        lo, hi = band
        return max(int(round(hi / sample_rate * n_fft)) - int(round(lo / sample_rate * n_fft)), 1)

    changed = True
    while changed:                      # split bands that are too wide for the tensor
        changed = False
        out: list[tuple[float, float]] = []
        for band in bands:
            if width_in_bins(band) > max_band_bins:
                lo, hi = band
                mid = 0.5 * (lo + hi)
                out.append((lo, mid))
                out.append((mid, hi))
                changed = True
            else:
                out.append(band)
        bands = out
    return bands


def bins_for_ranges(
    ranges_hz: list[tuple[float, float]], sample_rate: int, n_fft: int, n_bins: int
) -> list[tuple[int, int]]:
    """Quantise (lo_hz, hi_hz) ranges onto the FFT-bin axis of one resolution.

    The rounding gaps (and the content below `fmin`) are closed by extending each band to
    the end of the previous one, so the bands form a partition of `[0, n_bins)`: *every*
    bin is covered by at least one band and no frequency is silently zeroed.
    """
    spans: list[tuple[int, int]] = []
    for lo, hi in ranges_hz:
        start = min(int(round(lo / sample_rate * n_fft)), n_bins - 1)
        end = max(int(round(hi / sample_rate * n_fft)), start + 1)
        spans.append((start, min(end, n_bins)))
    spans[0] = (0, spans[0][1])         # clamp the edges ...
    filled: list[tuple[int, int]] = [spans[0]]
    for start, end in spans[1:]:        # ... and close the rounding gaps
        prev_end = filled[-1][1]
        filled.append((min(start, prev_end), max(end, prev_end + 1)))
    filled[-1] = (filled[-1][0], n_bins)
    return [(min(s, n_bins - 1), min(max(e, s + 1), n_bins)) for s, e in filled]


def build_bands(
    n_bins: int,
    sample_rate: int,
    n_fft: int,
    n_bands: int = 64,
    fmin: float = 30.0,
    fmax: float | None = None,
    band_overlap: float = 0.25,
    max_band_bins: int = 32,
) -> list[tuple[int, int]]:
    """Mel-spaced, overlapping, gap-free bin ranges covering `[0, n_bins)`.

    `band_hz_ranges` decides the bands (mel spacing, overlap, splitting of wide bands),
    `bins_for_ranges` puts them on this resolution's FFT grid.
    """
    ranges = band_hz_ranges(n_bands, fmin, fmax, band_overlap, sample_rate, n_fft,
                            max_band_bins)
    return bins_for_ranges(ranges, sample_rate, n_fft, n_bins)


def band_index_tensors(spans: list[tuple[int, int]], device=None):
    """Pad the ragged band ranges into `(index, valid)` tensors of shape (bands, Wmax).

    `index[b, k]` is the k-th FFT bin of band b (`0` where padding) and `valid[b, k]` says
    whether that entry is real. Gathering with `index` and masking with `valid` is what
    lets every band go through one batched matmul instead of a Python loop.
    """
    width = max(end - start for start, end in spans)
    index = torch.zeros((len(spans), width), dtype=torch.long, device=device)
    valid = torch.zeros((len(spans), width), dtype=torch.bool, device=device)
    for b, (start, end) in enumerate(spans):
        n = end - start
        index[b, :n] = torch.arange(start, end, device=device)
        valid[b, :n] = True
    return index, valid


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE, Su et al. 2021) over a sequence axis.

    Applied to bands it encodes how many bands apart two tokens are, which is a
    *frequency* distance; applied to frames it encodes time. Both are relative, so a model
    trained on 3 s chunks keeps working on the 430-frame windows of a whole song at
    inference time - there is no positional table that would need resizing.
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        assert dim % 2 == 0, "RoPE needs an even head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., seq, head_dim) -> rotated by its own position on the sequence axis."""
        seq = x.shape[-2]
        pos = torch.arange(seq, device=x.device, dtype=self.inv_freq.dtype)
        angles = torch.outer(pos, self.inv_freq)            # (seq, head_dim/2)
        cos, sin = angles.cos(), angles.sin()
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RoPESelfAttention(nn.Module):
    """Multi-head self-attention over the sequence axis of a (seq, B, N, dim) tensor.

    The layout is the one the blocks use: `seq` is the axis to attend over (bands *or*
    time), `B` is the audio batch and `N` is the remaining axis folded into the attention
    batch (frames *or* bands). Folding `N` is what makes the two attentions of a block
    cost about the same as a standard "batch over channels" transformer.
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, batch, n, dim = x.shape
        flat = x.permute(1, 2, 0, 3).reshape(batch * n, seq, dim)
        q, k, v = self.to_qkv(flat).chunk(3, dim=-1)
        shape = (batch * n, seq, self.heads, self.head_dim)
        q = q.reshape(shape).transpose(1, 2)                # (B*N, heads, seq, head_dim)
        k = k.reshape(shape).transpose(1, 2)
        v = v.reshape(shape).transpose(1, 2)
        q, k = self.rotary.rotate(q), self.rotary.rotate(k)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(batch * n, seq, dim)
        out = out.reshape(batch, n, seq, dim).permute(2, 0, 1, 3)
        return self.to_out(out)


class Block(nn.Module):
    """One RoFormer block: band attention, then time attention, then the feed-forward."""

    def __init__(self, dim: int, heads: int, ff_mult: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_bands = nn.LayerNorm(dim)
        self.band_attn = RoPESelfAttention(dim, heads, dropout)
        self.norm_time = nn.LayerNorm(dim)
        self.time_attn = RoPESelfAttention(dim, heads, dropout)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult * 2),
            nn.GLU(dim=-1),                                # gated units, as in the papers
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (bands, B, T, dim) -> same shape."""
        x = x + self.band_attn(self.norm_bands(x))          # sequence axis = bands
        t = x.permute(2, 1, 0, 3)                           # (T, B, bands, dim)
        t = t + self.time_attn(self.norm_time(t))           # sequence axis = time
        x = t.permute(2, 1, 0, 3)
        return x + self.ff(self.norm_ff(x))


class MelBandRoFormer(nn.Module):
    """Band-split RoFormer that predicts one complex mask per (source, channel, bin, frame).

        mixture spectrum (B, ch, F, T) complex
          -> gather the bins of every mel band                  (bands, B, T, 2*ch*Wmax)
          -> per-band linear + band embedding                   (bands, B, T, dim)
          + optional second-resolution log-magnitude features   (bands, B, T, ch)
          -> depth x (band attention + time attention + FFN)    (bands, B, T, dim)
          -> per-band mask head, tanh                          (bands, Wmax, B, sources, ch, 2, T)
          -> scatter the bands back onto the FFT grid, averaging the overlaps
          -> masks (B, sources, ch, F, T) complex

    The mask is a *ratio* mask (cIRM): `tanh` bounds it to [-1, 1], and the wrapper
    multiplies it by the mixture spectrum. `mask="sigmoid"` gives the real gain variant,
    matching the U-Net's two modes.
    """

    def __init__(
        self,
        n_bins: int,
        channels: int = 2,
        n_sources: int = 4,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        *,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        ff_mult: int = 4,
        n_bands: int = 64,
        band_overlap: float = 0.25,
        fmin: float = 30.0,
        fmax: float | None = None,
        max_band_bins: int = 32,
        n_fft_extra: int = 0,
        mask: str = "tanh",
        per_channel_mask: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.n_sources = n_sources
        self.n_bins = n_bins
        self.mask = mask
        self.per_channel_mask = per_channel_mask
        self.mask_channels = channels if per_channel_mask else 1
        self.extra_n_fft = n_fft_extra

        ranges = band_hz_ranges(n_bands, fmin, fmax, band_overlap, sample_rate, n_fft,
                                max_band_bins)
        index, valid = band_index_tensors(
            bins_for_ranges(ranges, sample_rate, n_fft, n_bins)
        )
        self.register_buffer("band_bins", index, persistent=False)
        self.register_buffer("band_valid", valid, persistent=False)
        self.n_bands, self.band_width = index.shape

        # every bin must belong to at least one band, otherwise that frequency would be
        # suppressed for *all* sources at once (the scatter below divides by this count)
        coverage = torch.zeros(n_bins)
        coverage.index_add_(0, index[valid], torch.ones(int(valid.sum())))
        assert float(coverage.min()) >= 1.0, "bands must cover the whole spectrum"
        self.register_buffer("bin_coverage", coverage, persistent=False)

        self.in_features = 2 * channels * self.band_width      # (re, im) x (L, R) x width
        self.in_proj = nn.Parameter(torch.empty(self.n_bands, self.in_features, dim))
        nn.init.normal_(self.in_proj, std=1.0 / math.sqrt(self.in_features))
        self.band_emb = nn.Parameter(torch.zeros(self.n_bands, dim))

        if n_fft_extra:
            extra_index, extra_valid = band_index_tensors(
                bins_for_ranges(ranges, sample_rate, n_fft_extra, n_fft_extra // 2 + 1)
            )
            assert extra_index.shape[0] == self.n_bands
            self.register_buffer("extra_bins", extra_index, persistent=False)
            self.register_buffer("extra_valid", extra_valid, persistent=False)
            self.extra_proj = nn.Parameter(torch.empty(self.n_bands, channels, dim))
            nn.init.normal_(self.extra_proj, std=1.0 / math.sqrt(channels))
        else:
            self.register_buffer("extra_bins", None, persistent=False)
            self.register_buffer("extra_valid", None, persistent=False)
            self.extra_proj = None

        self.blocks = nn.ModuleList(
            Block(dim, heads, ff_mult, dropout) for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.out_per_band = n_sources * self.mask_channels * 2 * self.band_width
        self.mask_head = nn.Linear(dim, self.out_per_band)
        self.mask_bias = nn.Parameter(torch.zeros(self.n_bands, self.out_per_band))

    # ------------------------------------------------------------------ features
    def band_features(self, mix_spec: torch.Tensor) -> torch.Tensor:
        """Complex mixture spectrum (B, ch, F, T) -> band tokens (bands, B, T, dim).

        The bands are gathered as raw (real, imaginary) pairs: unlike the U-Net, no
        log-magnitude "image" is built here, because the per-band linear layers can learn
        any monotone rescaling themselves. The only preprocessing is one scale factor per
        example (the RMS of the spectrum, detached from the graph), which keeps the
        numbers in a range a transformer likes without biasing the gradient.
        """
        x = torch.view_as_real(mix_spec)                        # (B, ch, F, T, 2)
        batch, ch, _, frames, _ = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(batch, 2 * ch, self.n_bins, frames)
        scale = x.detach().flatten(2).pow(2).mean(-1).sqrt().clamp_min(1e-6)
        x = x / scale[:, :, None, None]

        gathered = x[:, :, self.band_bins, :]                   # (B, 2ch, bands, Wmax, T)
        gathered = gathered * self.band_valid[None, None, :, :, None]
        feats = gathered.permute(2, 0, 4, 1, 3).reshape(
            self.n_bands, batch, frames, self.in_features
        )
        # per-band linear: `einsum` (not `@`) because the weight is stacked per band
        return torch.einsum("nbtf,nfd->nbtd", feats, self.in_proj) \
            + self.band_emb[:, None, None, :]

    def extra_features(self, extra_spec: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Second-resolution spectrogram (B, ch, F2, T2) -> (bands, B, T, ch).

        Only the log-magnitudes are used, band-pooled and standardised per band: this
        input exists to hand the model a second *view* of the same sound (shorter window =
        sharper attacks), not to add another mask. The frame axes of the two resolutions
        do not line up exactly, so `T2` is resampled to `T` by nearest neighbour - both
        cover the same time span, only the framing differs.
        """
        mag = torch.log(extra_spec.abs().clamp_min(1e-6))
        gathered = mag[:, :, self.extra_bins, :]                # (B, ch, bands, Wmax2, T2)
        gathered = gathered * self.extra_valid[None, None, :, :, None]
        count = self.extra_valid.sum(-1).clamp_min(1)[None, None, :, None]
        pooled = gathered.sum(dim=3) / count                    # (B, ch, bands, T2)
        mean = pooled.mean(dim=(0, 3), keepdim=True)
        std = pooled.std(dim=(0, 3), keepdim=True).clamp_min(1e-5)
        pooled = ((pooled - mean) / std).permute(2, 0, 3, 1)    # (bands, B, T2, ch)
        if pooled.shape[2] != n_frames:
            bands, batch, t2, ch = pooled.shape
            reshaped = pooled.permute(0, 3, 1, 2).reshape(bands * ch, batch, t2)
            pooled = F.interpolate(reshaped, size=n_frames, mode="nearest")
            pooled = pooled.reshape(bands, ch, batch, n_frames).permute(0, 2, 3, 1)
        return pooled

    # ---------------------------------------------------------------------- masks
    def masks_from_bands(self, h: torch.Tensor, n_frames: int, batch: int) -> torch.Tensor:
        """Band tokens (bands, B, T, dim) -> complex masks (B, sources, ch, F, T)."""
        out = self.mask_head(self.final_norm(h)) + self.mask_bias[:, None, None, :]
        out = out.reshape(self.n_bands, batch, n_frames, self.n_sources,
                          self.mask_channels, 2, self.band_width)
        out = out.permute(0, 6, 1, 3, 4, 5, 2)          # (bands, Wmax, B, S, cm, 2, T)
        if self.mask == "tanh":
            out = torch.tanh(out)
        elif self.mask == "sigmoid":
            out = torch.sigmoid(out)

        flat = out.reshape(self.n_bands * self.band_width, batch, self.n_sources,
                           self.mask_channels, 2, n_frames)
        valid_flat = self.band_valid.reshape(-1)
        acc = torch.zeros(self.n_bins, *flat.shape[1:], dtype=flat.dtype, device=flat.device)
        acc.index_add_(0, self.band_bins.reshape(-1)[valid_flat], flat[valid_flat])
        # overlapping bands predict the same bins twice: average, so no frequency is
        # weighted more than another just because its band happened to overlap more
        acc = acc / self.bin_coverage.clamp_min(1).reshape(
            -1, *([1] * (flat.dim() - 1))
        )
        real, imag = acc.unbind(dim=4)                          # (F, B, S, cm, T) each
        if self.mask == "sigmoid":
            imag = torch.zeros_like(imag)       # a real gain changes magnitude only
        masks = torch.complex(real, imag)
        masks = masks.permute(1, 2, 3, 0, 4)                    # (B, S, cm, F, T)
        if not self.per_channel_mask:
            masks = masks.expand(-1, -1, self.channels, -1, -1)
        return masks

    def forward(self, mix_spec: torch.Tensor,
                extra_spec: torch.Tensor | None = None) -> torch.Tensor:
        """(B, ch, F, T) complex mixture -> (B, sources, ch, F, T) complex masks."""
        batch, _, _, n_frames = mix_spec.shape
        h = self.band_features(mix_spec)
        if self.extra_proj is not None:
            if extra_spec is None:
                raise ValueError("this model was built with n_fft_extra > 0: "
                                 "pass the second-resolution spectrogram too")
            h = h + torch.einsum("nbtc,ncd->nbtd", self.extra_features(extra_spec, n_frames),
                                 self.extra_proj)
        for block in self.blocks:
            h = block(h)
        return self.masks_from_bands(h, n_frames, batch)
