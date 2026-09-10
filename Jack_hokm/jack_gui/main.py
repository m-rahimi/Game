from __future__ import annotations

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label

from .assets import suit_png
from .background_widget import Background
from .config import GuiConfig, parse_args, with_hokm
from .collection_widget import CollectionWidget
from .game_controller import GameController
from .hand_widget import PlayerHandWidget
from .layout import (
    configure_window,
    ROUND_BANNER_DURATION,
    TRUMP_MOVE_DURATION,
    TRUMP_REVEAL_HOLD,
)
from .opponent_widget import OpponentWidget
from .played_jacks_widget import PlayedJacksWidget
from .score_widget import ScoreWidget
from .trick_widget import TrickWidget


class GameBoard(FloatLayout):
    def __init__(self, config: GuiConfig, **kwargs):
        super().__init__(**kwargs)
        self.gui_config = config
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

        self.ai_status_label = Label(
            text="",
            font_size=14,
            bold=True,
            color=(1, 0.95, 0.4, 1),
            outline_color=(0, 0, 0, 0.85),
            outline_width=1,
            size_hint=(None, None),
            halign="left",
            valign="bottom",
            opacity=0,
        )
        self.ai_status_label.bind(size=self.ai_status_label.setter("text_size"))
        self.background.add_widget(self.ai_status_label)

        self.bind(size=self._on_viewport_change, pos=self._on_viewport_change)
        Window.bind(on_resize=self._on_viewport_change)

        self.controller = GameController(self, config)
        self.background.remove_image()
        Clock.schedule_once(self._on_viewport_change, 0)
        Clock.schedule_once(lambda _dt: self._begin_or_ask_hokm(), 0)

    def _begin_or_ask_hokm(self) -> None:
        if self.gui_config.ask_hokm and self.gui_config.hokm is None:
            self.show_hokm_dialog()
            return
        hokm = bool(self.gui_config.hokm)
        self.begin_game(hokm)

    def show_hokm_dialog(self) -> None:
        if self.overlay is not None:
            return
        self.clear_overlays()

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
            text="type of the game?",
            font_size=32,
            bold=True,
            color=(1, 1, 1, 1),
            size_hint=(1, 0.45),
            halign="center",
            valign="middle",
        )
        title.bind(size=title.setter("text_size"))

        buttons = BoxLayout(orientation="horizontal", size_hint=(1, 0.35), spacing=16)

        def _close_and_choose(hokm: bool) -> None:
            self.background.unbind(size=_sync_dim)
            self.background.unbind(pos=_sync_dim)
            self.clear_overlays()
            self.begin_game(hokm)

        with_hokm_btn = Button(
            text="with Hokm",
            font_size=26,
            bold=True,
            background_color=(0.25, 0.55, 0.95, 1),
            color=(1, 1, 1, 1),
        )
        without_hokm_btn = Button(
            text="without Hokm",
            font_size=26,
            bold=True,
            background_color=(0.92, 0.35, 0.35, 1),
            color=(1, 1, 1, 1),
        )
        with_hokm_btn.bind(on_press=lambda _btn: _close_and_choose(True))
        without_hokm_btn.bind(on_press=lambda _btn: _close_and_choose(False))
        buttons.add_widget(with_hokm_btn)
        buttons.add_widget(without_hokm_btn)

        panel.add_widget(title)
        panel.add_widget(buttons)
        overlay.add_widget(panel)
        self.overlay = overlay
        self.background.add_widget(overlay)

    def begin_game(self, hokm: bool) -> None:
        assert self.controller is not None
        self.gui_config = with_hokm(self.gui_config, hokm)
        self.controller.set_hokm(hokm)
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
        if self.controller and self.controller.config.ai == "policy":
            self.set_ai_status(self.controller.policy_status_text())

    def set_ai_status(self, text: str) -> None:
        visible = bool(text)
        if visible:
            self.ai_status_label.text = text
        else:
            self.ai_status_label.text = ""
        self.ai_status_label.opacity = 1 if visible else 0
        self._layout_ai_status()
        if visible:
            self._raise_ai_status()

    def _raise_ai_status(self) -> None:
        self.background.remove_widget(self.ai_status_label)
        self.background.add_widget(self.ai_status_label)

    def _layout_ai_status(self) -> None:
        from .layout import viewport_size

        width, height = viewport_size()
        if width <= 0 or height <= 0 or not self.ai_status_label.opacity:
            return
        font_size = max(14, height * 0.028)
        text_h = font_size * 1.4
        self.ai_status_label.font_size = font_size
        self.ai_status_label.size = (width * 0.14, text_h)
        self.ai_status_label.pos = (0, 0)
        self.ai_status_label.text_size = (self.ai_status_label.width, None)

    def clear_overlays(self):
        if self.overlay is not None:
            self.background.remove_widget(self.overlay)
            self.overlay = None

    def show_trump_suit(self, trump: str, callback) -> None:
        from .layout import viewport_size

        self.score_widget.hide_trump()
        self.score_widget.relayout()
        width, height = viewport_size()
        start_size = height * 0.28
        start_pos = (width / 2 - start_size / 2, height / 2 - start_size / 2)
        image = Image(
            source=suit_png(trump),
            size_hint=(None, None),
            size=(start_size, start_size),
            pos=start_pos,
            allow_stretch=True,
            keep_ratio=True,
        )
        self.background.add_widget(image)
        self.background.remove_widget(image)
        self.background.add_widget(image)
        self._trump_reveal_image = image

        def _start_move(_dt) -> None:
            end_size, end_pos = self.score_widget.trump_badge_layout_in_parent()
            anim = Animation(
                pos=end_pos,
                size=end_size,
                duration=TRUMP_MOVE_DURATION,
            )

            def _on_complete(_animation, _widget) -> None:
                self._finish_trump_reveal(trump, callback)

            anim.bind(on_complete=_on_complete)
            anim.start(image)

        Clock.schedule_once(_start_move, TRUMP_REVEAL_HOLD)

    def _finish_trump_reveal(self, trump: str, callback) -> None:
        if getattr(self, "_trump_reveal_image", None) is not None:
            self.background.remove_widget(self._trump_reveal_image)
            self._trump_reveal_image = None
        self.score_widget.show_trump(trump)
        if callback:
            callback()

    def show_round_end(self, tiebreak: bool = False):
        self.clear_overlays()
        if tiebreak:
            text = "Both teams at 21 — tie-break round"
        else:
            text = "Next round..."
        self._show_banner(text, duration=ROUND_BANNER_DURATION, callback=lambda: self.controller.start_round() if self.controller else None)

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
                if self.gui_config.ask_hokm:
                    self.show_hokm_dialog()
                else:
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
