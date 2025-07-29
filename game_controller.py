from kivy.animation import Animation
from card_widget import CardWidget
from kivy.clock import Clock
from kivy.core.window import Window

class GameController:
    def __init__(self, game_state, game_board):
        self.game_state = game_state  # Logic layer
        self.game_board = game_board  # UI layer
        self.win_cards, self.win_scores = [], []
        self.win_n = 0
        self.click_flag = True  # Prevent multiple clicks
        Window.bind(on_mouse_down=self.on_mouse_down)

    def on_card_clicked(self, card_widget):
        """Handle card click event"""
        if self.click_flag:
            self.click_flag = False  # Prevent further clicks until the current action is complete
            print("controller: Card clicked")
            self.game_board.player1_widget.remove_widget(card_widget)
            self.game_board.floor_widget.add_widget(card_widget)
            # Update game logic, add card to floor
            card_widget.name = 'Floor' 
            group = card_widget.group
            card = self.game_state.players[0].hand[group].pop(0)  # Remove the card from player's hand
            # find winning cards
            self.win_cards, self.win_scores = self.game_state.floor.find_winning_cards(card)
            self.game_state.floor.add_card(card)  # Add card to the floor

            if len(self.game_state.players[0].hand[group])==0:
                for card_none in self.game_board.player1_widget.children:
                    if isinstance(card_none, CardWidget) and card_none.card is None and card_none.group == group:
                        print("Removing empty card widget from player1")
                        self.game_board.player1_widget.remove_widget(card_none)
                        break            
            if self.win_cards:
                self.game_board.floor_widget.rearrange_card_widgets(card_widget, callback=self.show_wining_cards)
            else:
                self.game_board.floor_widget.rearrange_card_widgets(card_widget, callback=self.add_card_to_floor)


    def add_card_to_floor(self, card_widget):
        """Callback to add card to floor after animation"""
        print("Adding card to floor")
        group = card_widget.group
        if self.game_state.players[0].hand[group]:
            card = self.game_state.players[0].hand[group][0]
            card_widget = CardWidget(card, name="Player1", group=group)
            self.game_board.player1_widget.add_widget(card_widget)
            print(len(self.game_state.players[0].hand[group]))
            print(f"Card {card.rank} of {card.suit} is now in group {card_widget.name}")

    # def select_wining_cards(self, card_widget):
    #     """Select winning cards from the floor"""
    #     self.shows_wining_cards()      

    def show_wining_cards(self, card_widget):
        cards = self.win_cards[self.win_n]
        for card in cards:
            for idx, child in enumerate(self.game_board.floor_widget.children):
                if isinstance(child, CardWidget) and child.card == card:
                    child.pos = (child.pos[0], child.pos[1] + child.size[1] * 0.2)  # Move card up

    def on_mouse_down(self, *args):
        if len(self.win_cards) > 1:
            self.win_n += 1
            if self.win_n >= len(self.win_cards):
                self.win_n = 0
            print(f"Showing winning cards: {self.win_n}")
            self.game_board.floor_widget.rearrange_card_widgets(None)
            Clock.schedule_once(lambda dt: self.show_wining_cards(None), 0.5)
        else:
            print("Mouse down event detected")