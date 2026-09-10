"""Startup checks and per-move stats for Android/desktop policy AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kivy.logger import Logger


@dataclass
class PolicyAssetReport:
    model_path: str
    model_exists: bool
    model_bytes: int
    sidecar_exists: bool
    sidecar_bytes: int
    meta_exists: bool
    feature_dim: int | None
    embedded_weights: bool
    ready: bool
    message: str


@dataclass
class MoveStats:
    policy: int = 0
    heuristic: int = 0

    def record(self, source: str) -> None:
        if source == "policy":
            self.policy += 1
        elif source == "heuristic":
            self.heuristic += 1

    @property
    def total(self) -> int:
        return self.policy + self.heuristic

    def summary(self) -> str:
        if self.total == 0:
            return "AI moves: none yet"
        pct = 100.0 * self.policy / self.total
        return f"AI policy {self.policy}/{self.total} ({pct:.0f}%) | heuristic fallback {self.heuristic}"

    def short_label(self) -> str:
        if self.total == 0:
            return "AI --%"
        pct = round(100.0 * self.policy / self.total)
        return f"AI {pct}%"


def inspect_policy_assets(model_path: str | None) -> PolicyAssetReport:
    if not model_path:
        return PolicyAssetReport(
            model_path="",
            model_exists=False,
            model_bytes=0,
            sidecar_exists=False,
            sidecar_bytes=0,
            meta_exists=False,
            feature_dim=None,
            embedded_weights=False,
            ready=False,
            message="No policy checkpoint configured.",
        )

    path = Path(model_path)
    meta_path = path.with_suffix(path.suffix + ".json")
    sidecar = Path(f"{path}.data")
    model_exists = path.is_file()
    model_bytes = path.stat().st_size if model_exists else 0
    sidecar_exists = sidecar.is_file()
    sidecar_bytes = sidecar.stat().st_size if sidecar_exists else 0
    meta_exists = meta_path.is_file()

    feature_dim = None
    embedded_weights = False
    if meta_exists:
        import json

        meta = json.loads(meta_path.read_text())
        feature_dim = int(meta.get("feature_dim", 0)) or None
        embedded_weights = bool(meta.get("embedded_weights", False))

    if not model_exists:
        message = f"Missing model file: {path}"
        ready = False
    elif sidecar_exists and not embedded_weights and model_bytes < 100_000:
        message = (
            f"Model weights are external ({sidecar.name}, {sidecar_bytes} bytes) "
            f"but only the small graph file ({model_bytes} bytes) may be bundled."
        )
        ready = False
    elif model_bytes < 100_000 and not sidecar_exists:
        message = f"Model file looks too small ({model_bytes} bytes); weights may be missing."
        ready = False
    else:
        message = f"Policy model ready ({model_bytes // 1024} KB)."
        ready = True

    return PolicyAssetReport(
        model_path=str(path),
        model_exists=model_exists,
        model_bytes=model_bytes,
        sidecar_exists=sidecar_exists,
        sidecar_bytes=sidecar_bytes,
        meta_exists=meta_exists,
        feature_dim=feature_dim,
        embedded_weights=embedded_weights or model_bytes >= 100_000,
        ready=ready,
        message=message,
    )


def log_policy_assets(model_path: str | None) -> PolicyAssetReport:
    report = inspect_policy_assets(model_path)
    Logger.info("Jack5: policy path=%s", report.model_path)
    Logger.info(
        "Jack5: model=%s bytes=%s sidecar=%s meta=%s feature_dim=%s",
        report.model_exists,
        report.model_bytes,
        report.sidecar_bytes if report.sidecar_exists else "none",
        report.meta_exists,
        report.feature_dim,
    )
    if report.ready:
        Logger.info("Jack5: %s", report.message)
    else:
        Logger.warning("Jack5: %s", report.message)
    return report
