from kivy.uix.widget import Widget
from card_widget import CardWidget
from kivy.core.window import Window
from kivy.animation import Animation
from kivy.clock import Clock

class FloorWidget(Widget):
    def __init__(self, floor, controller, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        # Add CardWidgets for each card in player's hand
        for idx, card in enumerate(floor.hand):
            card_widget = CardWidget(card, name="Floor", group=idx)
            self.add_widget(card_widget)

    def rearrange_card_widgets(self, card_widget, callback=None):
        
        card_count = sum(1 for child in self.children if isinstance(child, CardWidget))
        size = self.children[0].size if self.children else (100, 150)  # Default size if no cards
        card_y = Window.height * 0.5 - size[1] / 2
        card_x = Window.width * 0.05
        card_x_max = card_x + (size[0] + Window.width * 0.01) * 5 
        if card_count <= 6:
            margin = Window.width * 0.01
        else:
            margin = (card_x_max - card_x) / (card_count-1) - size[0]

        # Animation for rearranging cards
        if card_widget:
            x_pos = card_x + (size[0] + margin) * (card_count - 1)
            anim = Animation(pos=(x_pos, card_y), duration=0.5)
            anim.bind(on_complete=lambda *args: callback(card_widget))
            anim.start(card_widget)
    
        # Update positions of all card widgets 
        for idx, child in enumerate(self.children):
            if isinstance(child, CardWidget):
                x_pos = card_x + (size[0] + margin) * (card_count - idx - 1)
                child.size = size
                child.pos = (x_pos, card_y)