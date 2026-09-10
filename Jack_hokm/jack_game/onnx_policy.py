"""Policy inference from an ONNX export of the PyTorch policy network."""

from __future__ import annotations

import json
from pathlib import Path
from random import Random

import numpy as np

from .bc import FEATURE_DIM_ATTR, FEATURE_DIM_HOKM, encode_state_from_game_state
from .cards import Card, SUITS, RANKS
from .onnx_runtime import OnnxModel

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def _metadata_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    return path.with_suffix(path.suffix + ".json")


def _load_metadata(model_path: str | Path) -> dict:
    meta_path = _metadata_path(model_path)
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    return {"feature_dim": FEATURE_DIM_ATTR, "hidden_dim": 256, "num_layers": 3}


_policy_players: dict[str, "GreedyOnnxPolicyPlayer"] = {}


def get_onnx_policy_player(model_path: str) -> "GreedyOnnxPolicyPlayer":
    path = str(model_path)
    player = _policy_players.get(path)
    if player is None:
        player = GreedyOnnxPolicyPlayer(path)
        _policy_players[path] = player
    return player


class OnnxPolicyFunction:
    def __init__(self, model_path: str, num_threads: int | None = 2) -> None:
        self.model_path = str(model_path)
        self.meta = _load_metadata(model_path)
        self.feature_dim = int(self.meta.get("feature_dim", FEATURE_DIM_ATTR))
        if self.feature_dim not in (FEATURE_DIM_ATTR, FEATURE_DIM_HOKM):
            raise ValueError(f"unsupported ONNX feature_dim={self.feature_dim}")
        self._android_model: OnnxModel | None = None
        self._desktop_session = None
        self._num_threads = num_threads

        if ort is not None:
            options = ort.SessionOptions()
            if num_threads is not None:
                options.intra_op_num_threads = num_threads
                options.inter_op_num_threads = 1
            self._desktop_session = ort.InferenceSession(self.model_path, options, providers=["CPUExecutionProvider"])

    def _android(self) -> OnnxModel:
        if self._android_model is None:
            self._android_model = OnnxModel(self.model_path, num_threads=self._num_threads)
        return self._android_model

    def warmup(self) -> None:
        if self._desktop_session is not None:
            return
        self._android()

    def _run_logits(self, features: list[float]) -> np.ndarray:
        batch = np.asarray([features], dtype=np.float32)
        if self._desktop_session is not None:
            outputs = self._desktop_session.run(None, {"features": batch})
            return outputs[0][0]
        return self._android().run({"features": batch}, {"logits": [52]})["logits"]

    def __call__(self, state, player: int) -> dict[int, float]:
        legal = state.legal_actions(player)
        legal_indexes = [card.index for card in legal]
        features, _ = encode_state_from_game_state(state, player, feature_dim=self.feature_dim)
        logits = self._run_logits(features)
        legal_logits = np.asarray([float(logits[index]) for index in legal_indexes], dtype=np.float32)
        legal_logits -= legal_logits.max()
        probs = np.exp(legal_logits)
        probs /= probs.sum()
        return {index: float(prob) for index, prob in zip(legal_indexes, probs)}


class GreedyOnnxPolicyPlayer:
    def __init__(self, model_path: str) -> None:
        self.policy_fn = OnnxPolicyFunction(model_path)

    def choose_card(self, state, player: int, rng: Random) -> Card:
        probs = self.policy_fn(state, player)
        best = max(probs.items(), key=lambda kv: kv[1])[0]
        return Card(SUITS[best // len(RANKS)], RANKS[best % len(RANKS)])
