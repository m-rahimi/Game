from __future__ import annotations

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label

from jack_game.cards import team_for
from jack_game.engine import GameState
from jack_game.session import MatchState

from .assets import icon_png
from .layout import round_score_layout, viewport_size

SCORE_FONT_RATIO = 0.09
SCORE_X_RATIO = 0.96
SCORE_BOX_WIDTH_RATIO = 0.08
SCORE_Y_TEAM1 = 0.70
SCORE_Y_TEAM0 = 0.30
ROUND_SCORE_FONT_SCALE = 0.5
MATCH_ICON_WIDTH_RATIO = 0.85


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
        self.add_widget(self.match_icon)

        self.ai_status_label = Label(
            text="",
            font_size=14,
            bold=True,
            color=(1, 0.95, 0.4, 1),
            size_hint=(None, None),
            halign="left",
            valign="top",
            opacity=0,
        )
        self.ai_status_label.bind(size=self.ai_status_label.setter("text_size"))
        self.add_widget(self.ai_status_label)

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
        center_y = height * (SCORE_Y_TEAM1 + SCORE_Y_TEAM0) / 2
        self.match_icon.size = (icon_w, icon_w)
        self.match_icon.pos = (x + (box_w - icon_w) / 2, center_y - icon_w / 2)

        if self.ai_status_label.opacity:
            status_w = width * 0.45
            self.ai_status_label.font_size = max(12, height * 0.022)
            self.ai_status_label.size = (status_w, height * 0.08)
            self.ai_status_label.pos = (width * 0.02, height * 0.02)

        round_font_size = font_size * ROUND_SCORE_FONT_SCALE
        for team, label in self.round_labels.items():
            size, pos = round_score_layout(team)
            label.font_size = round_font_size
            label.size = size
            label.pos = pos

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

    def set_ai_status(self, text: str) -> None:
        if text:
            self.ai_status_label.text = text
            self.ai_status_label.opacity = 1
        else:
            self.ai_status_label.text = ""
            self.ai_status_label.opacity = 0
        self._reposition()
