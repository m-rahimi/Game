from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Line
from kivy.core.window import Window
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.uix.image import Image
import os

ratio = 323 / 222
CARD_FOLDER = "png"
SUITS = ["clubs", "diamonds", "hearts", "spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace"]
RANKS_TO_NUMBER = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "ace": 1, "jack": 11, "queen": 12, "king": 13}

class CardWidget(Widget):
    def __init__(self, card, name, group, **kwargs):
        super().__init__(**kwargs)
        if card is None:
            filename = "png/blue.png"
        else:
            filename = f"{CARD_FOLDER}/{card.rank}_of_{card.suit}.png"
        self.card = card
        self.group = group
        self.name = name
        self.calculate_initial_pos_size_cards()
        with self.canvas:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)
            Color(0,0,0,1)
            self.border = Line(rectangle=(*self.pos, *self.size), width=1.2)
            self.card_image = Image(source=filename, size=self.size, pos=self.pos, allow_stretch=True, keep_ratio=True, mipmap=True)
        self.bind(pos=self.update_graphics, size=self.update_graphics)
        Clock.schedule_once(self.bind_to_parent)
        Clock.schedule_once(lambda dt: self.update_graphics())

    def update_graphics(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size
        self.border.rectangle = (*self.pos, *self.size)
        self.card_image.size = self.size
        self.card_image.pos = self.pos

    def bind_to_parent(self, *args):
        if self.parent:
            self.parent.bind(size=self.on_parent_resize)

    def on_parent_resize(self, instance, value):
        # Update size and position based on parent size
        self.calculate_initial_pos_size_cards()
        self.update_graphics()

    def on_window_resize(self, *args):
        # Recalculate positions only when the window size changes
        self.calculate_initial_pos_size_cards()

    def calculate_initial_pos_size_cards(self, *args):
        name = self.name
        width = Window.width * 0.1
        height = width * ratio
        if height > Window.height * 0.25:
            height = Window.height * 0.25
            width = height / ratio
        card_size = (int(width), int(height))

        if name == "Player1":
            card_x = Window.width * 0.05
            card_y = Window.height * 0.05
            card_margin = Window.width * 0.01
        elif name == "Computer":
            card_x = Window.width * 0.05
            card_y = Window.height * 0.95 - height
            card_margin = Window.width * 0.01
        elif name == "Floor":
            card_x = Window.width * 0.05
            card_y = Window.height * 0.5 - height / 2
            card_margin = Window.width * 0.01

        x_pos =  card_x + (width + card_margin) * self.group
        y_pos =  card_y

        self.size = card_size
        self.pos = (x_pos, y_pos)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if self.name == "Player1":
                if hasattr(self.parent.controller, 'on_card_clicked') and (self.parent.controller.click_flag):
                    self.parent.controller.on_card_clicked(self)
                return True
        return super().on_touch_down(touch)


