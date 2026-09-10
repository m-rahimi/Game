"""IIMC — information-consistent Monte Carlo action evaluation (Step 2D, 2026-08-28).

Why
---
2C showed the binding constraint is CREDIT ASSIGNMENT: one paired scalar smeared over
~41 decisions improved the sampled policy (+0.15 at temp 1.0) but left the argmax
ranking untouched (+0.04 ±0.095). To change *which* action is best in a state we need
per-action signal. This module produces it, acting as a policy-improvement operator
(the role MCTS plays in AlphaZero) rather than a gradient tweak.

IIMC vs PIMC — the distinction this whole project turns on
----------------------------------------------------------
For a decision by ``player`` we estimate, for each candidate action:

    Q(s, a) = E_{world ~ belief} [ team_diff after playing a, then letting the round
                                   finish with every seat acting on ITS OWN observation ]

Two properties matter, and both are deliberate:

* **Averaged over sampled worlds**, so the target does not depend on hidden cards the
  student cannot see. Conditioning on the true deal (which self-play knows) would
  produce an unachievable, oracle-flavoured target — the 2A failure mode.
* **Rollout seats act observable-only**, each on its own information set. This is what
  makes it IIMC and not PIMC. `players.py:127-151` scores candidates by rolling out a
  FULLY DETERMINIZED deal with `HeuristicPlayer`, so every simulated seat effectively
  knows all four hands — that is strategy fusion, and it is why the classical `search`
  player never values concealment or signalling. Here the rollout players are as
  uncertain as the real ones, so a line that is safe across worlds scores well.

So Q is the value of an action to an agent that *is* uncertain, which is exactly the
target an observable-only policy can actually attain — unlike the oracle's policy.

Cost
----
``|actions| x worlds`` rollouts per decision, each ~26 policy calls at ~1 ms. That is
2-3 orders of magnitude more expensive than 2C's data, so callers sample a few
decisions per game (``decisions_per_game``) and restrict to the policy's ``top_k``
candidates. Training-time only — inference stays one forward pass.
"""
from __future__ import annotations

from functools import lru_cache
from multiprocessing import Pool
from random import Random
from statistics import NormalDist

import torch

from .bc import FEATURE_DIM_ATTR, NUM_CARDS, encode_state_from_game_state
from .cards import Card
from .engine import GameState, new_round, play_card
from .exploit import ActorSpec, PolicyActor
from .players import sample_information_set


def iimc_action_returns(state: GameState, player: int, actor: PolicyActor, rng: Random,
                        worlds: int = 8, top_k: int = 4,
                        prior: dict[int, float] | None = None) -> dict[int, list[float]]:
    """-> {action index: [return in world 0, return in world 1, ...]}.

    The per-world detail that :func:`iimc_action_values` averages away. Because every
    candidate is played out on the SAME ``sampled_worlds`` (common random numbers) and
    the rollout seats act at argmax, ``returns[a][w] - returns[b][w]`` is an *exact*
    paired difference for world ``w``: the world's own luck cancels rather than adding
    noise. Callers that need to know how significant a difference is — the override
    rule, the training-row confidence weight — need these, not the means.
    """
    legal = state.legal_actions(player)
    if len(legal) <= 1:
        return {card.index: [] for card in legal}

    candidates = legal
    if top_k and top_k > 0 and len(legal) > top_k:
        if prior is None:
            _, _, _, _, _, prior_vector = actor.act_full(state, player, rng)
            prior = {card.index: prior_vector[card.index] for card in legal}
        candidates = sorted(legal, key=lambda card: -prior[card.index])[:top_k]

    # Same worlds for every candidate.
    sampled_worlds = [sample_information_set(state, player, rng) for _ in range(worlds)]

    returns: dict[int, list[float]] = {}
    for card in candidates:
        per_world = []
        for world in sampled_worlds:
            rollout = world.clone()
            play_card(rollout, player, card)
            _rollout_observable(rollout, actor, rng)
            per_world.append(float(rollout.team_diff_for(player)))
        returns[card.index] = per_world
    return returns


def iimc_action_values(state: GameState, player: int, actor: PolicyActor, rng: Random,
                       worlds: int = 8, top_k: int = 4,
                       prior: dict[int, float] | None = None) -> dict[int, float]:
    """-> {action index: Q}, the IIMC value of each evaluated action.

    ``top_k`` keeps only the highest-prior candidates (all of them if ``top_k <= 0``).
    Every candidate is evaluated on the SAME sampled worlds, which is a common-random-
    numbers comparison: world luck cancels between actions instead of adding noise to
    their difference.
    """
    returns = iimc_action_returns(state, player, actor, rng, worlds=worlds,
                                  top_k=top_k, prior=prior)
    return {action: (sum(values) / len(values) if values else 0.0)
            for action, values in returns.items()}


def _rollout_observable(state: GameState, actor: PolicyActor, rng: Random) -> None:
    """Finish the round with every seat acting on its own observation only.

    The seats see the sampled world's cards in their own hands and the public history —
    never each other's hands. That information consistency is the whole point.
    """
    while not state.round_over:
        seat = state.current_player
        legal = state.legal_actions(seat)
        if len(legal) == 1:
            play_card(state, seat, legal[0])
            continue
        card, *_ = actor.act(state, seat, rng)
        play_card(state, seat, card)


def override_choice(returns: dict[int, list[float]], anchor: int, margin: float = 0.0,
                    z: float = 0.0, bonferroni: bool = True,
                    floor: float = 0.0) -> tuple[int, float, float]:
    """Decide whether to override the prior's ``anchor`` choice. -> (action, mean d, SE d).

    Two rules, selected by ``z``:

    * ``z == 0`` — the original FIXED MARGIN: deviate iff the best mean beats the
      anchor's mean by ``margin``. One threshold for every decision, easy and hard alike.
    * ``z > 0`` — a PAIRED SIGNIFICANCE TEST on ``d_w = r_w(a) - r_w(anchor)``, which is
      an exact per-world difference (rollout seats play argmax, so a rollout is a
      deterministic function of (world, action) — verified by ``measure_paired_se.py``).

    Note what the test does and does not buy. Measured over 3000 real decisions at
    worlds=64 (``data/paired_se_w64.pt``), the paired SE is 0.412 against a marginal SE
    of 0.418 — common random numbers remove only ~3% of the variance, because playing a
    different card re-routes the whole remainder of the round. So this is NOT the
    variance reduction it looks like; it is a RECALIBRATION. Against the fixed margin it
    rejects ~269/3000 large-but-noisy deviations and accepts ~54 small-but-certain ones.

    ``bonferroni`` matters: taking the max over k-1 alternatives is a selection, so an
    uncorrected per-candidate test fires roughly k-1 times too often — that is the
    winner's curse re-entering through the door the margin was holding shut. ``floor``
    keeps an absolute minimum effect size, guarding the low-effective-n decisions (8.4%
    of comparisons have fewer than 8 worlds that separate the two actions at all).
    """
    if anchor not in returns:
        best = max(returns, key=lambda a: sum(returns[a]) / max(len(returns[a]), 1))
        return best, 0.0, 0.0
    base = returns[anchor]
    n = len(base)
    if n == 0:
        return anchor, 0.0, 0.0

    best, best_mean, best_se = anchor, 0.0, 0.0
    for action, values in returns.items():
        if action == anchor:
            continue
        diffs = [value - other for value, other in zip(values, base)]
        mean = sum(diffs) / n
        if mean <= best_mean:
            continue
        if n > 1:
            variance = sum((d - mean) ** 2 for d in diffs) / (n - 1)
            se = (variance / n) ** 0.5
        else:
            se = float("inf")
        best, best_mean, best_se = action, mean, se

    if best == anchor:
        return anchor, 0.0, 0.0
    if z <= 0.0:
        return (best if best_mean > margin else anchor), best_mean, best_se
    alternatives = max(len(returns) - 1, 1)
    z_effective = _bonferroni_z(z, alternatives) if bonferroni else z
    threshold = max(z_effective * best_se, floor)
    return (best if best_mean > threshold else anchor), best_mean, best_se


@lru_cache(maxsize=64)
def _bonferroni_z(z: float, comparisons: int) -> float:
    """The z that keeps the FAMILY-wise error at ``z``'s nominal level across
    ``comparisons`` candidates. Bonferroni splits alpha, so the correction is on the
    tail probability (alpha -> alpha/k), not on z itself: at alpha=0.05 and k=3 this is
    1.645 -> 2.128, whereas scaling z by sqrt(k) would give a far too conservative
    2.85."""
    if comparisons <= 1:
        return z
    alpha = 1.0 - NormalDist().cdf(z)
    return NormalDist().inv_cdf(1.0 - alpha / comparisons)


class IIMCPlayer:
    """Plays argmax of the IIMC action values. TRAINING/DIAGNOSTIC ONLY — it runs
    rollouts, so it violates the single-forward-pass deployment constraint. Its job is
    to answer "are the improved targets actually better than the policy they came
    from?" before we spend compute distilling them."""

    def __init__(self, actor: PolicyActor, worlds: int = 8, top_k: int = 4,
                 margin: float = 0.0, z: float = 0.0, bonferroni: bool = True,
                 floor: float = 0.0) -> None:
        self.actor = actor
        self.worlds = worlds
        self.top_k = top_k
        # Q estimates from `worlds` rollouts carry a standard error of roughly
        # 4.66/sqrt(worlds) points, while real action differences are fractions of a
        # point. Taking a raw argmax therefore selects whichever action drew the
        # luckiest worlds (winner's curse) and measurably DAMAGES play (-1.015
        # pts/round at worlds=8, 2026-08-28). `margin` requires Q to beat the policy's
        # own top choice by that much before overriding it, which is the same job PUCT
        # does in MCTS: trust the prior until the value evidence is strong.
        self.margin = margin
        # z > 0 swaps that fixed threshold for a per-decision paired test; see
        # `override_choice` for what it is and is not worth.
        self.z = z
        self.bonferroni = bonferroni
        self.floor = floor

    def choose(self, state: GameState, player: int, rng: Random) -> Card:
        legal = state.legal_actions(player)
        if len(legal) == 1:
            return legal[0]
        _, _, _, _, _, prior = self.actor.act_full(state, player, rng)
        prior_map = {card.index: prior[card.index] for card in legal}
        returns = iimc_action_returns(state, player, self.actor, rng,
                                      worlds=self.worlds, top_k=self.top_k,
                                      prior=prior_map)
        anchor = max(prior_map.items(), key=lambda item: item[1])[0]
        choice, _, _ = override_choice(returns, anchor, margin=self.margin, z=self.z,
                                       bonferroni=self.bonferroni, floor=self.floor)
        return _card(choice)


def _card(index: int) -> Card:
    from .mcts import index_to_card
    return index_to_card(index)


class ActorPlayer:
    """Adapts a ``PolicyActor`` to the same ``choose`` interface as ``IIMCPlayer``."""

    def __init__(self, actor: PolicyActor) -> None:
        self.actor = actor

    def choose(self, state: GameState, player: int, rng: Random) -> Card:
        card, *_ = self.actor.act(state, player, rng)
        return card


# --------------------------------------------------------------------------
# paired-deal evaluation for arbitrary players (the early gate)
# --------------------------------------------------------------------------

def paired_round(deal_seed: int, leader: int, player_a, player_b, a_team: int,
                 rng: Random, hokm: bool = False) -> float:
    """Play one round with ``player_a`` on ``a_team``; return a's team diff."""
    state = new_round(seed=deal_seed, leader=leader, hokm=hokm)
    seats = {a_team, a_team + 2}
    while not state.round_over:
        seat = state.current_player
        chooser = player_a if seat in seats else player_b
        play_card(state, seat, chooser.choose(state, seat, rng))
    return float(state.team_diff_for(a_team))


def paired_eval(player_a, player_b, deal_seeds, rng: Random, hokm: bool = False) -> dict:
    """Duplicate-deal scoring for arbitrary players: each deal played twice with the
    seats swapped, ``paired = d_a + d_b``. Same algebra as jack_game/exploit.py, but
    accepting players that are not plain nets."""
    paired_values, diffs = [], []
    wins = draws = losses = 0
    for index, seed in enumerate(deal_seeds):
        leader = seed % 4
        d_a = paired_round(seed, leader, player_a, player_b, 0, rng, hokm=hokm)
        d_b = paired_round(seed, leader, player_a, player_b, 1, rng, hokm=hokm)
        paired_values.append(d_a + d_b)
        for diff in (d_a, d_b):
            diffs.append(diff)
            wins += int(diff > 0)
            draws += int(diff == 0)
            losses += int(diff < 0)
    pairs = len(paired_values)
    mean = sum(paired_values) / pairs
    variance = max(sum(v * v for v in paired_values) / pairs - mean * mean, 0.0)
    playings = len(diffs)
    return {
        "pairs": pairs, "playings": playings,
        "edge": mean / 2.0,
        "edge_se": (variance ** 0.5) / (2.0 * pairs ** 0.5),
        "score_per_game": sum(diffs) / playings,
        "win_rate": (wins + 0.5 * draws) / playings,
        "wins": wins, "draws": draws, "losses": losses,
    }


def _eval_chunk(arg):
    (seeds, seed, actor_spec, opponent_spec, worlds, top_k, use_iimc, margin,
     z, bonferroni, floor) = arg
    torch.set_num_threads(1)
    rng = Random(seed)
    actor = PolicyActor(actor_spec)
    player_a = (IIMCPlayer(actor, worlds=worlds, top_k=top_k, margin=margin,
                           z=z, bonferroni=bonferroni, floor=floor)
                if use_iimc else ActorPlayer(actor))
    player_b = ActorPlayer(PolicyActor(opponent_spec))
    return paired_eval(player_a, player_b, seeds, rng)


def _merge_eval(chunks: list[dict]) -> dict:
    pairs = sum(c["pairs"] for c in chunks)
    playings = sum(c["playings"] for c in chunks)
    # Recombine the per-chunk means/variances into pooled statistics.
    total = sum(c["edge"] * 2.0 * c["pairs"] for c in chunks)
    mean = total / pairs
    second = sum(((c["edge_se"] * 2.0 * c["pairs"] ** 0.5) ** 2
                  + (c["edge"] * 2.0) ** 2) * c["pairs"] for c in chunks) / pairs
    variance = max(second - mean * mean, 0.0)
    wins = sum(c["wins"] for c in chunks)
    draws = sum(c["draws"] for c in chunks)
    losses = sum(c["losses"] for c in chunks)
    return {
        "pairs": pairs, "playings": playings,
        "edge": mean / 2.0,
        "edge_se": (variance ** 0.5) / (2.0 * pairs ** 0.5),
        "score_per_game": sum(c["score_per_game"] * c["playings"] for c in chunks) / playings,
        "win_rate": (wins + 0.5 * draws) / playings,
        "wins": wins, "draws": draws, "losses": losses,
    }


def parallel_paired_eval(actor_spec: ActorSpec, opponent_spec: ActorSpec, pairs: int,
                         workers: int, seed: int, worlds: int = 8, top_k: int = 4,
                         use_iimc: bool = True, margin: float = 0.0, z: float = 0.0,
                         bonferroni: bool = True, floor: float = 0.0) -> dict:
    """Evaluate the IIMC player (or the bare net, ``use_iimc=False``) against a frozen
    opponent on ``pairs`` duplicate deals."""
    rng = Random(seed)
    all_seeds = [rng.randrange(10 ** 9) for _ in range(pairs)]
    per = [all_seeds[i::workers] for i in range(workers)]
    tasks = [(chunk, seed + 7919 * i, actor_spec, opponent_spec, worlds, top_k,
              use_iimc, margin, z, bonferroni, floor)
             for i, chunk in enumerate(per) if chunk]
    if len(tasks) == 1:
        return _eval_chunk(tasks[0])
    results = []
    with Pool(processes=len(tasks)) as pool:
        for result in pool.imap_unordered(_eval_chunk, tasks):
            results.append(result)
    return _merge_eval(results)


# --------------------------------------------------------------------------
# improved-target data generation
# --------------------------------------------------------------------------

def _data_chunk(arg):
    (games, seed, game_base, actor_spec, opponent_spec, worlds, top_k,
     decisions_per_game, play_temp, margin) = arg
    torch.set_num_threads(1)
    rng = Random(seed)
    actor = PolicyActor(actor_spec)
    play_actor = PolicyActor(ActorSpec(
        state_dict=actor_spec.state_dict, feature_dim=actor_spec.feature_dim,
        hidden_dim=actor_spec.hidden_dim, num_layers=actor_spec.num_layers,
        temperature=play_temp, oracle=actor_spec.oracle))
    opponent = PolicyActor(opponent_spec)

    feats: list[list[float]] = []
    targets: list[list[float]] = []
    qs: list[list[float]] = []
    legals: list[list[bool]] = []
    game_ids: list[int] = []

    for index in range(games):
        team = index % 2
        seats = {team, team + 2}
        state = new_round(seed=rng.randrange(10 ** 9), leader=index % 4)
        # First pass: collect this game's non-forced decision points for our seats.
        decisions: list[tuple[GameState, int]] = []
        while not state.round_over:
            seat = state.current_player
            legal = state.legal_actions(seat)
            if len(legal) == 1:
                play_card(state, seat, legal[0])
                continue
            if seat in seats:
                decisions.append((state.clone(), seat))
                card, *_ = play_actor.act(state, seat, rng)
            else:
                card, *_ = opponent.act(state, seat, rng)
            play_card(state, seat, card)

        if not decisions:
            continue
        rng.shuffle(decisions)
        for snapshot, seat in decisions[:decisions_per_game]:
            _, _, _, _, _, prior = actor.act_full(snapshot, seat, rng)
            prior_map = {card.index: prior[card.index]
                         for card in snapshot.legal_actions(seat)}
            values = iimc_action_values(snapshot, seat, actor, rng, worlds=worlds,
                                       top_k=top_k, prior=prior_map)
            if len(values) <= 1:
                continue
            features, illegal = encode_state_from_game_state(
                snapshot, seat, feature_dim=FEATURE_DIM_ATTR)
            q_vector = [0.0] * NUM_CARDS
            target = [0.0] * NUM_CARDS
            for action, value in values.items():
                q_vector[action] = value
            # Target = the MARGIN-ANCHORED choice, i.e. exactly the operator that was
            # verified to beat its own policy. A raw argmax-Q target would bake in the
            # winner's curse that made worlds=8 play 1.0 pts/round WORSE than the net.
            anchor = max(prior_map.items(), key=lambda item: item[1])[0]
            best, best_value = max(values.items(), key=lambda item: item[1])
            if anchor in values and best_value < values[anchor] + margin:
                best = anchor
            target[best] = 1.0
            feats.append(features)
            qs.append(q_vector)
            targets.append(target)
            # Only the EVALUATED actions are a valid support for the target; actions we
            # never rolled out must not be treated as legal-but-rejected.
            legals.append([action in values for action in range(NUM_CARDS)])
            game_ids.append(game_base + index)

    return (
        torch.tensor(feats, dtype=torch.float16),
        torch.tensor(targets, dtype=torch.float16),
        torch.tensor(qs, dtype=torch.float16),
        torch.tensor(legals, dtype=torch.bool),
        torch.tensor(game_ids, dtype=torch.int32),
    )


def generate_iimc_data(actor_spec: ActorSpec, opponent_spec: ActorSpec, games: int,
                       workers: int, seed: int, worlds: int = 8, top_k: int = 4,
                       decisions_per_game: int = 4, play_temp: float = 0.5,
                       margin: float = 0.5) -> dict:
    """Play ``games`` rounds, then IIMC-evaluate ``decisions_per_game`` sampled decisions
    from each. Records observable features, the argmax-Q one-hot target, the raw Q
    vector (so a trainer can build soft targets or fit a Q head) and a game id."""
    per = [games // workers] * workers
    for i in range(games - sum(per)):
        per[i] += 1
    tasks, base = [], 0
    for count in per:
        if count > 0:
            tasks.append((count, seed + 7919 * len(tasks), base, actor_spec,
                          opponent_spec, worlds, top_k, decisions_per_game, play_temp,
                          margin))
            base += count

    parts = []
    if len(tasks) == 1:
        parts.append(_data_chunk(tasks[0]))
    else:
        with Pool(processes=len(tasks)) as pool:
            for result in pool.imap_unordered(_data_chunk, tasks):
                parts.append(result)

    keys = ("features", "target", "q", "legal", "game_id")
    data: dict[str, torch.Tensor] = {}
    for position, key in enumerate(keys):
        pieces = [part[position] for part in parts if part[position].numel() > 0]
        data[key] = torch.cat(pieces) if pieces else torch.empty(0)
    return data
