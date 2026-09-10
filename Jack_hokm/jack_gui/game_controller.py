from __future__ import annotations

import threading
from random import Random

from kivy.clock import Clock
from kivy.logger import Logger
from kivy.utils import platform as kivy_platform

from jack_game.cards import team_for
from jack_game.cli import make_ai
from jack_game.engine import GameState, TrickResult, new_round, play_card
from jack_game.players import HeuristicPlayer
from jack_game.session import MATCH_TARGET, MatchState

from .card_widget import CardWidget
from .config import (
    GuiConfig,
    effective_policy_checkpoint,
    effective_value_checkpoint,
    with_hokm,
)
from .ai_status import MoveStats, log_policy_assets
from .layout import display_seat, AI_MOVE_DELAY, TRICK_FINISH_DELAY, ROUND_END_DELAY


class GameController:
    def __init__(self, board, config: GuiConfig):
        self.board = board
        self.config = config
        self.human_seat = config.human_seat
        self.seed = config.seed
        self.rng = Random(config.seed)
        self.match = MatchState.new(self.rng)
        self.state: GameState | None = None
        self.bots = []
        self.click_enabled = True
        self.ai_delay = AI_MOVE_DELAY
        self._ai_busy = False
        self._heuristic_fallback = HeuristicPlayer()
        self._move_stats = MoveStats()
        self._policy_report = None
        self._log = None
        self.hokm = bool(config.hokm) if config.hokm is not None else False
        if config.log_path:
            from jack_game.gamelog import GameLog

            self._log = GameLog(
                config.log_path,
                meta={
                    "ai": config.ai,
                    "human_player": config.human_seat,
                    "simulations": config.simulations,
                    "checkpoint": config.checkpoint,
                    "policy_checkpoint": config.policy_checkpoint,
                    "value_checkpoint": config.value_checkpoint,
                    "belief_checkpoint": config.belief_checkpoint,
                    "seed": config.seed,
                    "match_target": MATCH_TARGET,
                    "hokm": self.hokm,
                },
            )
            Logger.info("Jack: logging rounds to %s", config.log_path)

    def set_hokm(self, hokm: bool) -> None:
        self.hokm = hokm
        self.config = with_hokm(self.config, hokm)
        self.board.score_widget.set_hokm_game(hokm)

    def _active_policy_checkpoint(self) -> str | None:
        return effective_policy_checkpoint(self.config, self.hokm)

    def _active_value_checkpoint(self) -> str | None:
        return effective_value_checkpoint(self.config, self.hokm)

    def _make_bot(self):
        cfg = self.config
        return make_ai(
            cfg.ai,
            rollouts=cfg.rollouts,
            lambda_clubs=cfg.lambda_clubs,
            lambda_hearts=cfg.lambda_hearts,
            time_limit=cfg.time_limit,
            checkpoint_path=cfg.checkpoint,
            policy_checkpoint=self._active_policy_checkpoint(),
            value_checkpoint=self._active_value_checkpoint(),
            simulations=cfg.simulations,
            belief_checkpoint=cfg.belief_checkpoint,
        )

    def start_match(self):
        self.match = MatchState.new(self.rng)
        self.start_round()

    def start_round(self):
        self.match.begin_round()
        round_seed = self.rng.randrange(10**9)
        self.state = new_round(seed=round_seed, leader=self.match.next_leader, hokm=self.hokm)
        self.bots = [self._make_bot() for _ in range(4)]
        self.click_enabled = False
        self._ai_busy = False
        self.board.clear_overlays()
        self.board.collection_widget.clear()
        self.board.trick_widget.clear()
        self.board.played_jacks_widget.clear()
        self.refresh()
        if self._log:
            self._log.start_game(self.state)
        self._policy_report = log_policy_assets(self._active_policy_checkpoint())
        self._warmup_onnx()
        if self.config.ai == "policy":
            self.board.set_ai_status(self.policy_status_text())
        if self.hokm and self.state.trump is not None:
            self.board.show_trump_suit(self.state.trump, self._begin_round_play)
        else:
            self.board.score_widget.hide_trump()
            self._begin_round_play()

    def _begin_round_play(self, _dt=None) -> None:
        assert self.state is not None
        self.click_enabled = True
        if self.state.current_player != self.human_seat:
            Clock.schedule_once(self._ai_move, self.ai_delay)

    def refresh(self):
        assert self.state is not None
        legal = set()
        if self.state.current_player == self.human_seat and not self.state.round_over:
            legal = set(self.state.legal_actions(self.human_seat))
        self.board.player_hand.refresh(self.state.hands[self.human_seat], legal)
        self.board.opponent_widget.refresh(self.state.hands, self.human_seat)
        self.board.score_widget.update(self.state, self.human_seat, self.match)
        self.board.played_jacks_widget.refresh(self.state)

    def on_card_clicked(self, widget: CardWidget):
        if not self.click_enabled or self.state is None or self.state.round_over:
            return
        if self.state.current_player != self.human_seat:
            return
        card = widget.card
        if card is None or card not in self.state.legal_actions(self.human_seat):
            return

        self.click_enabled = False
        if self._log:
            self._log.record_move(self.human_seat, card, human=True)
        result = play_card(self.state, self.human_seat, card)
        self.board.played_jacks_widget.refresh(self.state)
        self.board.player_hand.detach_widget(widget)
        self.board.opponent_widget.refresh(self.state.hands, self.human_seat)

        display = display_seat(self.human_seat, self.human_seat)

        def after_anim():
            self._after_play(self.human_seat, result)

        self.board.trick_widget.animate_from_hand(widget, display, after_anim)

    def policy_status_text(self) -> str:
        report = self._policy_report
        if report is not None and not report.ready:
            return "AI !"
        return self._move_stats.short_label()

    def _warmup_onnx(self) -> None:
        if kivy_platform != "android" or self.config.ai != "policy":
            return
        ckpt = self._active_policy_checkpoint()
        if not ckpt or not str(ckpt).endswith(".onnx"):
            return
        try:
            from jack_game.onnx_policy import get_onnx_policy_player

            get_onnx_policy_player(str(ckpt)).policy_fn.warmup()
        except Exception as exc:
            Logger.warning("Jack: ONNX warmup failed: %s", exc)

    def _ai_move(self, _dt):
        assert self.state is not None
        if self.state.round_over or self._ai_busy:
            return
        seat = self.state.current_player
        if seat == self.human_seat:
            self.click_enabled = True
            self.refresh()
            return

        self.click_enabled = False
        self._ai_busy = True
        bot = self.bots[seat]
        state_snapshot = self.state.clone()

        def compute(_dt=None):
            source = "policy"
            try:
                card, visits, q = self._choose_ai_card(bot, state_snapshot, seat)
            except Exception as exc:
                source = "heuristic"
                self._handle_ai_error(seat, exc)
                return
            self._record_ai_move(seat, card, source, visits=visits, q=q)
            self._apply_ai_move(seat, card)

        # JNI / ONNX Runtime must run on Android's main thread; background threads hang.
        if kivy_platform == "android":
            Clock.schedule_once(compute, 0)
        else:
            def compute_thread():
                source = "policy"
                try:
                    card, visits, q = self._choose_ai_card(bot, state_snapshot, seat)
                except Exception as exc:
                    Clock.schedule_once(lambda _dt: self._handle_ai_error(seat, exc), 0)
                    return
                Clock.schedule_once(
                    lambda _dt: self._record_and_apply(seat, card, source, visits=visits, q=q),
                    0,
                )

            threading.Thread(target=compute_thread, daemon=True).start()

    def _choose_ai_card(self, bot, state, seat: int):
        if self._log and hasattr(bot, "choose_with_stats"):
            card, visits, q = bot.choose_with_stats(state, seat, self.rng)
            return card, visits, q
        return bot.choose_card(state, seat, self.rng), None, None

    def _record_and_apply(self, seat: int, card, source: str, visits=None, q=None):
        self._record_ai_move(seat, card, source, visits=visits, q=q)
        self._apply_ai_move(seat, card)

    def _record_ai_move(self, seat: int, card, source: str, visits=None, q=None) -> None:
        if self._log:
            self._log.record_move(seat, card, human=False, visits=visits, q=q)
        self._move_stats.record(source)
        Logger.info("Jack: seat %s played %s via %s | %s", seat, card, source, self._move_stats.summary())
        if self.config.ai == "policy":
            self.board.set_ai_status(self.policy_status_text())

    def _handle_ai_error(self, seat: int, exc: Exception):
        Logger.exception("Jack: policy inference failed for seat %s, using heuristic: %s", seat, exc)
        self._ai_busy = False
        assert self.state is not None
        if self.state.round_over or self.state.current_player != seat:
            return
        card = self._heuristic_fallback.choose_card(self.state.clone(), seat, self.rng)
        self._record_ai_move(seat, card, "heuristic", visits=None, q=None)
        self._apply_ai_move(seat, card)

    def _apply_ai_move(self, seat: int, card):
        self._ai_busy = False
        assert self.state is not None
        if self.state.round_over or self.state.current_player != seat:
            return

        result = play_card(self.state, seat, card)
        self.board.played_jacks_widget.refresh(self.state)
        self.board.opponent_widget.refresh(self.state.hands, self.human_seat)

        display = display_seat(seat, self.human_seat)

        def after_anim():
            self._after_play(seat, result)

        self.board.trick_widget.animate_from_opponent(card, display, after_anim)

    def _after_play(self, seat: int, result: TrickResult | None):
        assert self.state is not None
        if result is not None:
            Clock.schedule_once(lambda _dt: self._finish_trick(result), TRICK_FINISH_DELAY)
            return
        if self.state.round_over:
            Clock.schedule_once(lambda _dt: self._end_round(), ROUND_END_DELAY)
            return
        if self.state.current_player != self.human_seat:
            Clock.schedule_once(self._ai_move, self.ai_delay)
        else:
            self.refresh()
            self.click_enabled = True

    def _finish_trick(self, result: TrickResult):
        assert self.state is not None
        winning_team = team_for(result.winner)

        def after_collect():
            if self.state.round_over:
                self._end_round()
                return
            self.refresh()
            if self.state.current_player != self.human_seat:
                Clock.schedule_once(self._ai_move, self.ai_delay)
            else:
                self.click_enabled = True

        self.board.trick_widget.collect_trick(
            winning_team,
            self.board.collection_widget,
            after_collect,
        )

    def _end_round(self):
        assert self.state is not None
        if self._log:
            self._log.end_game(self.state)
        self.click_enabled = False
        self.board.trick_widget.clear()
        self.board.collection_widget.clear()

        result = self.match.apply_round(self.state)
        self.board.score_widget.update(self.state, self.human_seat, self.match)
        if result is None:
            self.match.rotate_leader()
            self.board.show_round_end()
            return

        if result == "tiebreak":
            self.match.tiebreak = True
            self.match.rotate_leader()
            self.board.show_round_end(tiebreak=True)
            return

        winner = int(result)
        human_team = team_for(self.human_seat)
        self.board.show_match_end(winner, human_team, self.match)
