from __future__ import annotations

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label

from jack_game.cards import team_for
from jack_game.engine import GameState
from jack_game.session import MatchState

from .assets import icon_png, suit_png
from .layout import round_score_layout, viewport_size

SCORE_FONT_RATIO = 0.09
SCORE_X_RATIO = 0.96
SCORE_BOX_WIDTH_RATIO = 0.08
# Match totals: fixed X on the right; Y at 35% / 50% / 65%.
SCORE_Y_TEAM0 = 0.35
SCORE_Y_CENTER = 0.50
SCORE_Y_TEAM1 = 0.65
ROUND_SCORE_FONT_SCALE = 0.5
MATCH_ICON_WIDTH_RATIO = 1.55
MATCH_ICON_ASPECT = 768 / 1408
MATCH_ICON_Y_OFFSET_RATIO = 0.04
TRUMP_BADGE_WIDTH_RATIO = 0.38
TRUMP_BADGE_GAP_RATIO = 0.10


class ScoreWidget(FloatLayout):
    def __init__(self, **kwargs):
        super().__init__(size_hint=(1, 1), **kwargs)
        self.match_labels: dict[int, Label] = {}
        self.round_labels: dict[int, Label] = {}
        for team in (0, 1):
            match_label = Label(
                text="00",
                font_size=48,
                bold=True,
                color=(1, 1, 1, 1),
                size_hint=(None, None),
                halign="center",
                valign="middle",
            )
            match_label.bind(size=match_label.setter("text_size"))
            self.match_labels[team] = match_label
            self.add_widget(match_label)

            round_label = Label(
                text="",
                font_size=24,
                bold=True,
                color=(1, 1, 1, 0.85),
                size_hint=(None, None),
                halign="center",
                valign="middle",
                opacity=0,
            )
            round_label.bind(size=round_label.setter("text_size"))
            self.round_labels[team] = round_label
            self.add_widget(round_label)

        self.match_icon = Image(
            source=icon_png(),
            size_hint=(None, None),
            allow_stretch=True,
            keep_ratio=True,
        )
        self.trump_icon = Image(
            source="",
            size_hint=(None, None),
            allow_stretch=True,
            keep_ratio=True,
            opacity=0,
        )
        self.add_widget(self.match_icon)
        self.add_widget(self.trump_icon)
        self._hokm_game = False

    def set_hokm_game(self, hokm: bool) -> None:
        self._hokm_game = hokm
        if not hokm:
            self.hide_trump()
        self._reposition()

    def relayout(self, *_args):
        self._reposition()

    def _reposition(self, *_args):
        width, height = viewport_size()
        if width <= 0 or height <= 0:
            return
        box_w = width * SCORE_BOX_WIDTH_RATIO
        box_h = box_w * 1.1
        font_size = height * SCORE_FONT_RATIO
        x = width * SCORE_X_RATIO - box_w
        y_by_team = {
            1: height * SCORE_Y_TEAM1 - box_h / 2,
            0: height * SCORE_Y_TEAM0 - box_h / 2,
        }
        for team, label in self.match_labels.items():
            label.font_size = font_size
            label.size = (box_w, box_h)
            label.pos = (x, y_by_team[team])

        icon_w = box_w * MATCH_ICON_WIDTH_RATIO
        icon_h = icon_w * MATCH_ICON_ASPECT
        center_y = height * SCORE_Y_CENTER
        if self._hokm_game:
            center_y -= height * MATCH_ICON_Y_OFFSET_RATIO
        self.match_icon.size = (icon_w, icon_h)
        self.match_icon.pos = (x + (box_w - icon_w) / 2, center_y - icon_h / 2)

        if self.trump_icon.opacity:
            badge_size, badge_pos = self.trump_badge_layout()
            self.trump_icon.size = badge_size
            self.trump_icon.pos = badge_pos

        round_font_size = font_size * ROUND_SCORE_FONT_SCALE
        for team, label in self.round_labels.items():
            size, pos = round_score_layout(team)
            label.font_size = round_font_size
            label.size = size
            label.pos = pos

    def trump_badge_layout(self) -> tuple[tuple[float, float], tuple[float, float]]:
        icon_w, icon_h = self.match_icon.size
        badge_w = icon_w * TRUMP_BADGE_WIDTH_RATIO
        badge_h = badge_w
        gap = icon_h * TRUMP_BADGE_GAP_RATIO
        badge_x = self.match_icon.x + (icon_w - badge_w) / 2
        badge_y = self.match_icon.y + icon_h + gap
        return (badge_w, badge_h), (badge_x, badge_y)

    def trump_badge_layout_in_parent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        size, (local_x, local_y) = self.trump_badge_layout()
        return size, (self.x + local_x, self.y + local_y)

    def show_trump(self, suit: str) -> None:
        self.trump_icon.source = suit_png(suit)
        size, pos = self.trump_badge_layout()
        self.trump_icon.size = size
        self.trump_icon.pos = pos
        self.trump_icon.opacity = 1

    def hide_trump(self) -> None:
        self.trump_icon.opacity = 0
        self.trump_icon.source = ""

    @staticmethod
    def _format_match_score(score: int) -> str:
        if score < 0:
            return str(score)
        return f"{min(score, 99):02d}"

    @staticmethod
    def _format_round_score(score: int) -> str:
        if score < 0:
            return str(score)
        return f"{min(score, 99):02d}"

    @staticmethod
    def _tricks_won(state: GameState, team: int) -> int:
        return sum(1 for result in state.completed_tricks if team_for(result.winner) == team)

    def update(self, state: GameState, human_seat: int, match: MatchState | None = None):
        if match is None:
            match_totals = {0: 0, 1: 0}
        else:
            match_totals = {0: match.team_totals[0], 1: match.team_totals[1]}
        round_totals = {0: state.team_score(0), 1: state.team_score(1)}

        for team, label in self.match_labels.items():
            label.text = self._format_match_score(match_totals[team])
        for team, label in self.round_labels.items():
            if self._tricks_won(state, team) == 0:
                label.text = ""
                label.opacity = 0
            else:
                label.text = self._format_round_score(round_totals[team])
                label.opacity = 1
        self._reposition()
