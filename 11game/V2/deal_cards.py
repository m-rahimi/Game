from kivy.uix.widget import Widget
from card_widget import CardWidget
from kivy.core.window import Window
from kivy.animation import Animation
from kivy.core.audio import SoundLoader
from kivy.clock import Clock

class deal_cards(Widget):
    def __init__(self, player1, player2, floor, callback, **kwargs):
        super().__init__(**kwargs)
        self.player1 = player1
        self.player2 = player2
        self.floor = floor
        self.group = 5
        self.name = "Player1"
        self.count = 0
        self.callback = callback

        self.sound = SoundLoader.load('sounds/card.mp3')

        card_widget_b = CardWidget(None, name="Floor", group=0)
        self.add_widget(card_widget_b)

        self.init_pos = card_widget_b.pos
        Clock.schedule_once(lambda dt: self.deal_players_cards(), 0.5)

    def deal_players_cards(self):
        if self.count < 12:
            if self.name == "Player1":
                card = self.player1.hand[self.group][0]
                self.move_card(card)
                self.name = "Computer"
                self.count += 1
            elif self.name == "Computer":
                card = self.player2.hand[self.group][0]
                self.move_card(card)
                self.name = "Player1"
                self.count += 1
            if self.count % 2 == 0:
                self.group -= 1
        else:
            self.group = 3
            self.name = "Floor"
            self.deal_floor_cards()

    def deal_floor_cards(self):
        if self.group >= 0:
            card = self.floor.hand[self.group]
            self.move_card(card)
            self.group -= 1
        else:
            self.sound.stop()
            self.callback()  # Notify that dealing is done
            
    def move_card(self, card):
        card_widget_b = CardWidget(None, name="Floor", group=0)
        card_widget = CardWidget(card, name=self.name, group=self.group)

        self.add_widget(card_widget_b)
        anim = Animation(pos=card_widget.pos, duration=0.1) #self.sound.length)
        anim.bind(on_complete=lambda *args: self.add_widget(card_widget))
        if self.name != "Floor":
            anim.bind(on_complete=lambda *args: self.deal_players_cards())
        else:
            anim.bind(on_complete=lambda *args: self.deal_floor_cards())
        anim.start(card_widget_b)
#        self.sound.play()


