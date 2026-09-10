from __future__ import annotations

from kivy.graphics import Color, Rectangle
from kivy.uix.floatlayout import FloatLayout

from .assets import card_background_png


class Background(FloatLayout):
    """Same felt table as 11game/backgroung_widget.py."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            Color(25 / 255, 124 / 255, 84 / 255, 1)
            self.background = Rectangle(size=self.size, pos=self.pos)
        with self.canvas:
            Color(1, 1, 1, 1)
            self.image = Rectangle(source=card_background_png(), size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

    def _update_rect(self, instance, _value):
        self.background.size = instance.size
        self.background.pos = instance.pos
        self.image.size = instance.size
        self.image.pos = instance.pos

    def remove_image(self):
        self.canvas.remove(self.image)
