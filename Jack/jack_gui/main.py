from __future__ import annotations

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label

from .background_widget import Background
from .config import GuiConfig, parse_args
from .collection_widget import CollectionWidget
from .game_controller import GameController
from .hand_widget import PlayerHandWidget
from .layout import configure_window
from .opponent_widget import OpponentWidget
from .played_jacks_widget import PlayedJacksWidget
from .score_widget import ScoreWidget
from .trick_widget import TrickWidget


class GameBoard(FloatLayout):
    def __init__(self, config: GuiConfig, **kwargs):
        super().__init__(**kwargs)
        self.controller: GameController | None = None
        self.overlay = None

        self.background = Background(size=self.size, pos=self.pos)
        self.add_widget(self.background)

        self.score_widget = ScoreWidget()
        self.played_jacks_widget = PlayedJacksWidget()
        self.collection_widget = CollectionWidget()
        self.trick_widget = TrickWidget()
        self.opponent_widget = OpponentWidget()
        self.player_hand = PlayerHandWidget()

        self.background.add_widget(self.score_widget)
        self.background.add_widget(self.played_jacks_widget)
        self.background.add_widget(self.collection_widget)
        self.background.add_widget(self.trick_widget)
        self.background.add_widget(self.opponent_widget)
        self.background.add_widget(self.player_hand)

        self.bind(size=self._on_viewport_change, pos=self._on_viewport_change)
        Window.bind(on_resize=self._on_viewport_change)

        self.controller = GameController(self, config)
        self.background.remove_image()
        Clock.schedule_once(self._on_viewport_change, 0)
        self.controller.start_match()

    def _on_viewport_change(self, *_args):
        self.background.size = self.size
        self.background.pos = self.pos
        if self.width <= 0 or self.height <= 0:
            return
        self.score_widget.relayout()
        self.played_jacks_widget.relayout()
        self.collection_widget.relayout()
        self.trick_widget.relayout()
        self.opponent_widget.relayout()
        self.player_hand.relayout()
        if self.controller and self.controller.config.debug_ai:
            self.score_widget.set_ai_status(self.controller._policy_status_text())

    def clear_overlays(self):
        if self.overlay is not None:
            self.background.remove_widget(self.overlay)
            self.overlay = None

    def show_round_end(self, tiebreak: bool = False):
        self.clear_overlays()
        if tiebreak:
            text = "Both teams at 21 — tie-break round"
        else:
            text = "Next round..."
        self._show_banner(text, duration=1.2, callback=lambda: self.controller.start_round() if self.controller else None)

    def show_match_end(self, winner: int, human_team: int, match):
        self.clear_overlays()
        won = winner == human_team
        overlay = FloatLayout(size_hint=(1, 1))

        def _sync_dim(instance, _value):
            overlay._dim.size = instance.size
            overlay._dim.pos = instance.pos

        with overlay.canvas.before:
            Color(0, 0, 0, 0.6)
            overlay._dim = Rectangle(size=self.background.size, pos=self.background.pos)
        self.background.bind(size=_sync_dim, pos=_sync_dim)

        panel = BoxLayout(
            orientation="vertical",
            size_hint=(0.55, 0.32),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            spacing=24,
            padding=24,
        )
        with panel.canvas.before:
            Color(0.04, 0.22, 0.12, 0.98)
            panel._bg = Rectangle(size=panel.size, pos=panel.pos)
        panel.bind(size=lambda i, v: setattr(panel._bg, "size", i.size))
        panel.bind(pos=lambda i, v: setattr(panel._bg, "pos", i.pos))

        title = Label(
            text="You Win!" if won else "You Lose!",
            font_size=32,
            bold=True,
            color=(1, 1, 1, 1),
            size_hint=(1, 0.45),
            halign="center",
            valign="middle",
        )
        title.bind(size=title.setter("text_size"))

        buttons = BoxLayout(orientation="horizontal", size_hint=(1, 0.35), spacing=16)

        def _on_new_game(_instance):
            self.background.unbind(size=_sync_dim)
            self.background.unbind(pos=_sync_dim)
            self.clear_overlays()
            if self.controller:
                self.controller.start_match()

        def _on_exit_game(_instance):
            App.get_running_app().stop()

        new_game_btn = Button(
            text="New Game",
            font_size=26,
            bold=True,
            background_color=(0.25, 0.55, 0.95, 1),
            color=(1, 1, 1, 1),
        )
        exit_game_btn = Button(
            text="Exit Game",
            font_size=26,
            bold=True,
            background_color=(0.92, 0.35, 0.35, 1),
            color=(1, 1, 1, 1),
        )
        new_game_btn.bind(on_press=_on_new_game)
        exit_game_btn.bind(on_press=_on_exit_game)
        buttons.add_widget(new_game_btn)
        buttons.add_widget(exit_game_btn)

        panel.add_widget(title)
        panel.add_widget(buttons)
        overlay.add_widget(panel)
        self.overlay = overlay
        self.background.add_widget(overlay)

    def _show_banner(self, text: str, duration: float, callback):
        label = Label(
            text=text,
            font_size=22,
            color=(1, 1, 1, 1),
            size_hint=(0.8, 0.15),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            halign="center",
            valign="middle",
        )
        label.bind(size=label.setter("text_size"))
        self.overlay = label
        self.background.add_widget(label)
        Clock.schedule_once(lambda _dt: self._finish_banner(callback), duration)

    def _finish_banner(self, callback):
        self.clear_overlays()
        if callback:
            callback()


class JackApp(App):
    def __init__(self, gui_config: GuiConfig, **kwargs):
        super().__init__(**kwargs)
        self.gui_config = gui_config

    def build(self):
        configure_window()
        return GameBoard(self.gui_config)


def run_app(config: GuiConfig):
    JackApp(config).run()


def main(argv: list[str] | None = None):
    config = parse_args(argv)
    run_app(config)


if __name__ == "__main__":
    main()
