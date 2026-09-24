#!/usr/bin/env python3
"""Resumable, threaded downloader for MUSDB18 (and the other SigSep Zenodo archives).

Zenodo throttles each connection heavily, so an archive is split into fixed-size
chunks that several threads fetch concurrently with HTTP Range requests. Completed
chunks are recorded in a small JSON sidecar, so the job can be interrupted (Ctrl-C,
sleep, crash) and restarted with the very same command: it resumes where it stopped.

Usage
-----
    python scripts/download_musdb.py --out data/musdb18.zip --extract-to data/musdb18
    python scripts/download_musdb.py --hq --out data/musdb18hq.zip --extract-to data/musdb18hq

The archive itself is never deleted, so extraction can be re-run offline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

MUSDB18_URL = "https://zenodo.org/records/1117372/files/musdb18.zip"
MUSDB18HQ_URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip"

USER_AGENT = "Mozilla/5.0 (compatible; musdb-fetch/1.0)"


def _request(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0):
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, headers=hdrs)
    return urllib.request.urlopen(req, timeout=timeout)


def remote_size(url: str) -> int:
    """Content length of `url`, using a 1-byte ranged GET (Zenodo redirects on HEAD)."""
    with _request(url, {"Range": "bytes=0-0"}) as resp:
        content_range = resp.headers.get("Content-Range")
        if content_range and "/" in content_range:
            return int(content_range.split("/")[-1])
        return int(resp.headers["Content-Length"])


@dataclass
class Progress:
    """Thread-safe progress bookkeeping shared by the download workers."""

    total: int
    done: int = 0
    failed: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._start = time.time()

    def add(self, nbytes: int) -> None:
        with self._lock:
            self.done += nbytes

    def add_failed(self) -> None:
        with self._lock:
            self.failed += 1

    def report(self) -> None:
        with self._lock:
            done, failed, elapsed = self.done, self.failed, time.time() - self._start
        rate = done / max(elapsed, 1e-9)
        pct = 100.0 * done / max(self.total, 1)
        eta = (self.total - done) / rate / 60 if rate > 0 else float("inf")
        print(
            f"[download] {pct:5.1f}%  {done / 1e6:8.1f}/{self.total / 1e6:.1f} MB  "
            f"{rate / 1e6:6.3f} MB/s  eta {eta:6.1f} min  failed_chunks={failed}",
            flush=True,
        )


def download(
    url: str,
    out_path: Path,
    threads: int = 8,
    chunk_bytes: int = 4 << 20,
    max_retries: int = 6,
) -> Path:
    """Download `url` to `out_path`, resuming any previously completed chunks."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_path.with_suffix(out_path.suffix + ".state.json")

    total = remote_size(url)
    n_chunks = (total + chunk_bytes - 1) // chunk_bytes
    piece_sizes = [min(chunk_bytes, total - i * chunk_bytes) for i in range(n_chunks)]

    state: dict = {"url": url, "size": total, "chunk_bytes": chunk_bytes, "done": []}
    if state_path.exists():
        try:
            previous = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            previous = {}
        if (
            previous.get("size") == total
            and previous.get("chunk_bytes") == chunk_bytes
        ):
            state = previous
        else:
            print("[download] state does not match archive, starting over", flush=True)

    done_chunks = {int(i) for i in state.get("done", [])}
    pending = [i for i in range(n_chunks) if i not in done_chunks]

    fd = os.open(out_path, os.O_CREAT | os.O_WRONLY, 0o644)
    os.ftruncate(fd, total)
    progress = Progress(total=total, done=sum(piece_sizes[i] for i in done_chunks))

    print(
        f"[download] {url}\n[download] -> {out_path} ({total / 1e6:.1f} MB, "
        f"{n_chunks} chunks of {chunk_bytes / 1e6:.1f} MB, {len(pending)} pending, "
        f"{threads} threads)",
        flush=True,
    )
    if not pending:
        progress.report()
        os.close(fd)
        print("[download] already complete", flush=True)
        return out_path

    lock = threading.Lock()
    queue = list(pending)
    flush_counter = [0]

    def flush_state() -> None:
        tmp = state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(state_path)

    def worker() -> None:
        while True:
            with lock:
                if not queue:
                    return
                idx = queue.pop()
            start = idx * chunk_bytes
            end = start + piece_sizes[idx] - 1
            for attempt in range(max_retries):
                try:
                    with _request(
                        url, {"Range": f"bytes={start}-{end}"}, timeout=180.0
                    ) as resp:
                        data = resp.read()
                    if len(data) != piece_sizes[idx]:
                        raise IOError(
                            f"short read: {len(data)} != {piece_sizes[idx]} bytes"
                        )
                    os.pwrite(fd, data, start)
                except Exception as exc:  # noqa: BLE001 - any network hiccup is retryable
                    wait = min(2**attempt, 30)
                    print(
                        f"[download] chunk {idx} attempt {attempt + 1} failed "
                        f"({exc!r}); retrying in {wait}s",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                progress.add(len(data))
                with lock:
                    done_chunks.add(idx)
                    state["done"] = sorted(done_chunks)
                    flush_counter[0] += 1
                    if flush_counter[0] % 16 == 0:
                        flush_state()
                break
            else:
                progress.add_failed()
                with lock:
                    queue.append(idx)

    t0 = time.time()
    for round_index in range(6):
        if not queue:
            break
        if round_index:
            print(f"[download] round {round_index + 1}: {len(queue)} chunks to retry",
                  flush=True)
        workers = [threading.Thread(target=worker, daemon=True) for _ in range(threads)]
        for w in workers:
            w.start()
        while any(w.is_alive() for w in workers):
            time.sleep(20)
            progress.report()
        for w in workers:
            w.join()
        flush_state()
        progress.report()
    os.close(fd)

    missing = total - sum(piece_sizes[i] for i in done_chunks)
    if missing > 0:
        print(
            f"[download] INCOMPLETE: {missing} bytes missing - re-run to resume",
            file=sys.stderr,
            flush=True,
        )
        return out_path
    print(f"[download] COMPLETE in {(time.time() - t0) / 60:.1f} min", flush=True)
    state_path.unlink(missing_ok=True)
    return out_path

def extract(zip_path: Path, dest: Path) -> None:
    """Extract `zip_path` into `dest`, skipping members that are already there."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        todo = [n for n in names if not (dest / n).exists()]
        print(f"[extract] {len(todo)}/{len(names)} members missing in {dest}", flush=True)
        for i, name in enumerate(todo, 1):
            zf.extract(name, dest)
            if i % 200 == 0 or i == len(todo):
                print(f"[extract] {i}/{len(todo)}", flush=True)


def verify_archive(zip_path: Path) -> bool:
    """Cheap integrity check: the central directory must be readable."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            print(f"[verify] {zip_path}: {len(zf.namelist())} members, OK", flush=True)
        return True
    except zipfile.BadZipFile as exc:
        print(f"[verify] {zip_path}: BROKEN ({exc})", file=sys.stderr, flush=True)
        return False


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=MUSDB18_URL,
                   help="archive URL (default: MUSDB18 compressed, 4.7 GB)")
    p.add_argument("--hq", action="store_true",
                   help="shortcut for MUSDB18-HQ (uncompressed wav, 22.7 GB)")
    p.add_argument("--out", type=Path, default=Path("data/musdb18.zip"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--chunk-mb", type=float, default=4.0)
    p.add_argument("--extract-to", type=Path, default=None)
    p.add_argument("--no-download", action="store_true",
                   help="only extract/verify an archive already on disk")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    url = MUSDB18HQ_URL if args.hq else args.url
    chunk_bytes = int(args.chunk_mb * (1 << 20))

    if not args.no_download:
        state_path = args.out.with_suffix(args.out.suffix + ".state.json")
        # A leftover state file means the download was interrupted: the sparse archive
        # already has its final *size*, so only the state file tells us it is unfinished.
        needs_download = state_path.exists() or not (
            args.out.exists() and args.out.stat().st_size == remote_size(url)
        )
        if needs_download:
            download(url, args.out, threads=args.threads, chunk_bytes=chunk_bytes)
        else:
            print(f"[download] {args.out} already complete", flush=True)
        if not verify_archive(args.out):
            return 1

    if args.extract_to is not None:
        extract(args.out, args.extract_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

