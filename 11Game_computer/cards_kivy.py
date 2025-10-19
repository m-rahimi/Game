from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
import os

CARD_WIDTH = 80 *2
CARD_HEIGHT = 120*2
MARGIN = 10
CARD_FOLDER = "png"

class Card(Widget):
    def __init__(self, x, y, filename, **kwargs):
        super().__init__(size_hint=(800, 600), **kwargs)  # <--- Add size_hint
        self.size = (CARD_WIDTH, CARD_HEIGHT)
        self.pos = (x, y)
        with self.canvas:
            Color(1, 1, 1, 1)  # White background
            self.bg_rect = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._update_graphics, size=self._update_graphics)

    def _update_graphics(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size
        # Recalculate image position if card moves
        x, y = self.pos
        img_width, img_height = self.img.size
        self.img.pos = (x + (CARD_WIDTH - img_width) / 2, y + (CARD_HEIGHT - img_height) / 2)

class CardGame(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Green background
        with self.canvas.before:
            Color(59/255, 122/255, 87/255, 1)  # RGB values for a green color
            self.bg = Rectangle(pos=self.pos, size=Window.size)
        self.bind(size=self._update_bg, pos=self._update_bg)

        self.player1_cards = []
        self.player2_cards = []
        self.floor_cards = []

        # Example card names (replace with your actual filenames)
        card_names = [
            "2_of_clubs.png", "3_of_hearts.png", "4_of_spades.png", "5_of_diamonds.png",
            "6_of_clubs.png", "7_of_hearts.png", "8_of_spades.png", "9_of_diamonds.png",
            "10_of_clubs.png", "jack_of_hearts.png", "queen_of_spades.png", "king_of_diamonds.png"
        ]

        # Player 1 cards (top)
        for i in range(6):
            group = []
            for j in range(4):
                x = 70 + i * (CARD_WIDTH + MARGIN)
                y = 400 + j * 10
                filename = card_names[(i*4 + j) % len(card_names)]
                card = Card(x, y, os.path.join(CARD_FOLDER, filename))
                self.add_widget(card)
                group.append(card)
            self.player1_cards.append(group)

        # Player 2 cards (bottom)
        for i in range(6):
            group = []
            for j in range(4):
                x = 70 + i * (CARD_WIDTH + MARGIN)
                y = 80 + j * 10
                filename = card_names[((i*4 + j + 12) % len(card_names))]
                card = Card(x, y, os.path.join(CARD_FOLDER, filename))
                self.add_widget(card)
                group.append(card)
            self.player2_cards.append(group)

        # Floor cards (middle)
        for i in range(4):
            x = 200 + i * (CARD_WIDTH + MARGIN)
            y = 250
            filename = card_names[(i + 24) % len(card_names)]
            card = Card(x, y, os.path.join(CARD_FOLDER, filename))
            self.add_widget(card)
            self.floor_cards.append(card)

    def _update_bg(self, *args):
        self.bg.pos = self.pos
        self.bg.size = self.size

class CardGameApp(App):
    def build(self):
        Window.size = (800, 600)
        return CardGame()

if __name__ == "__main__":
    CardGameApp().run()