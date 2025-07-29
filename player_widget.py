from kivy.uix.widget import Widget
from card_widget import CardWidget

class PlayerWidget(Widget):
    def __init__(self, player, name, controller, **kwargs):
        super().__init__(**kwargs)
        self.player = player
        self.controller = controller
        self.name = name
        # Add CardWidgets for each card in player's hand
        for idx, cards in enumerate(player.hand):
            if cards:
                card = None #cards[0]
                card_widget = CardWidget(card, name=self.name, group=idx)
                self.add_widget(card_widget)
                card = cards[0]
                card_widget = CardWidget(card, name=self.name, group=idx)
                self.add_widget(card_widget)                

    def remove_card_widget(self, card):
        for child in self.children:
            if isinstance(child, CardWidget) and child.card == card:
                self.remove_widget(child)
                break

    def on_card_clicked(self, card_widget):
        print(f"Card clicked: {card_widget.card}")
        self.controller.on_card_clicked(card_widget)  # Notify controller about the card click