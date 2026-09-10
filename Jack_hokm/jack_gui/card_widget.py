from __future__ import annotations

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Line, PopMatrix, PushMatrix, Rectangle, Rotate
from kivy.properties import NumericProperty
from kivy.uix.image import Image
from kivy.uix.widget import Widget

from jack_game.cards import Card

from .card_adapter import card_png_path
from .layout import CARD_ANIM_DURATION, card_size


class CardWidget(Widget):
    """Same rendering as 11game/card_widget.py, with Jack card PNG paths."""

    rotation = NumericProperty(0)

    def __init__(
        self,
        card: Card | None,
        name: str = "Player1",
        group: int = 0,
        playable: bool = False,
        rotation: float = 0,
        **kwargs,
    ):
        super().__init__(size_hint=(None, None), **kwargs)
        self.card = card
        self.name = name
        self.group = group
        self.playable = playable
        self.rotation = rotation
        self._apply_size_and_pos()
        source = card_png_path(card)
        with self.canvas:
            PushMatrix()
            self.rot = Rotate(angle=self.rotation, origin=(0, 0))
            Color(1, 1, 1, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)
            self.card_image = Image(
                source=source,
                size=self.size,
                pos=self.pos,
                allow_stretch=True,
                keep_ratio=True,
                mipmap=True,
            )
            Color(1, 1, 1, 1)
            border_width = 2.5 if playable else 1.2
            self.border = Line(rectangle=(*self.pos, *self.size), width=border_width)
            PopMatrix()
        self.bind(pos=self.update_graphics, size=self.update_graphics, rotation=self.update_graphics)
        Clock.schedule_once(lambda _dt: self.update_graphics())

    def _apply_size_and_pos(self):
        width, height = card_size()
        if self.name == "Player1":
            self.size = (width, height)
            self.pos = (Window.width * 0.05, Window.height * 0.05)
        elif self.name == "Computer":
            card_x = Window.width * 0.05
            card_y = Window.height * 0.95 - height
            card_margin = Window.width * 0.01
            self.size = (width, height)
            self.pos = (card_x + (width + card_margin) * self.group, card_y)
        elif self.name == "Floor":
            card_x = Window.width * 0.05
            card_y = Window.height * 0.5 - height / 2
            card_margin = Window.width * 0.01
            self.size = (width, height)
            self.pos = (card_x + (width + card_margin) * self.group, card_y)
        elif self.name == "Trick":
            from .layout import trick_card_position

            self.size = (width, height)
            _, self.pos = trick_card_position(self.group)
        else:
            self.size = (width, height)
            self.pos = (Window.width * 0.05, Window.height * 0.05)

    def set_card_face(self, card: Card):
        self.card = card
        self.card_image.source = card_png_path(card)
        self.update_graphics()

    def set_card_back(self):
        from .assets import card_back_png

        self.card_image.source = card_back_png()
        self.update_graphics()

    def update_graphics(self, *_args):
        ox = self.x + self.width / 2
        oy = self.y + self.height / 2
        self.rot.angle = self.rotation
        self.rot.origin = (ox, oy)
        self.rect.pos = self.pos
        self.rect.size = self.size
        border_width = max(1.0, self.width * (0.022 if self.playable else 0.011))
        self.border.width = border_width
        self.border.rectangle = (*self.pos, *self.size)
        self.card_image.pos = self.pos
        self.card_image.size = self.size

    def cancel_animations(self):
        Animation.cancel_all(self, "pos", "rotation")

    def snap_layout(self, pos, size=None, rotation=None):
        self.cancel_animations()
        if size is not None:
            self.size = size
        self.pos = pos
        if rotation is not None:
            self.rotation = rotation
        self.update_graphics()

    def animate_to(self, pos, rotation=None, duration=CARD_ANIM_DURATION, callback=None):
        if rotation is None:
            anim = Animation(pos=pos, duration=duration)
        else:
            anim = Animation(pos=pos, rotation=rotation, duration=duration)
        if callback:
            anim.bind(on_complete=lambda *_a: callback())
        anim.start(self)

    def set_playable(self, playable: bool):
        self.playable = playable
        self.border.width = 2.5 if playable else 1.2

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        if self.name != "Player1":
            return super().on_touch_down(touch)
        board = self._find_board()
        if board and board.controller and board.controller.click_enabled:
            board.controller.on_card_clicked(self)
            return True
        return super().on_touch_down(touch)

    def _find_board(self):
        node = self.parent
        while node is not None:
            if hasattr(node, "controller"):
                return node
            node = getattr(node, "parent", None)
        return None
