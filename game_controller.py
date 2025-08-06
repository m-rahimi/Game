from kivy.animation import Animation
from card_widget import CardWidget
from kivy.clock import Clock
from kivy.core.window import Window
import copy
import random

class GameController:
    def __init__(self, game_state, game_board):
        self.game_state = game_state  # Logic layer
        self.game_board = game_board  # UI layer
        self.win_cards, self.win_scores = [], []
        self.win_n = 0
        self.click_flag = True  # Prevent multiple clicks
        self.card_played1_group = None
        self.card_played2_group = None
        self.computer_flag = False
        self.last_winner = None  # Track the last winner
        self.soor = False  # Track if a SOOOR has occurred
        self.player1_soor = 0
        self.player2_soor = 0

    def on_card_clicked(self, card_widget):
        """Handle card click event for Player 1"""
        if self.click_flag and self.computer_flag is False:
            self.click_flag = False  # Prevent further clicks until the current action is complete
            self.card_played1_group = copy.deepcopy(card_widget.group)
            self.game_board.player1_widget.remove_widget(card_widget)
            self.game_board.floor_widget.add_widget(card_widget)

            # Update game logic, add card to floor
            card_widget.name = 'Floor' 
            group = card_widget.group
            card = self.game_state.players[0].hand[group].pop(0)  # Remove the card from player's hand

            # find winning cards
            self.win_cards, self.win_scores, self.soor = self.game_state.floor.find_winning_cards(card)
            self.game_state.floor.add_card(card)  # Add card to the floor
            if len(self.win_scores)>0: #self.soor:
                self.player1_soor += 1
                print(f"Player 1 SOOOR count: {self.player1_soor}")
                self.game_board.show_soor.update_soors(self.player1_soor, self.player2_soor)

            # remove empty card widget from player1
            if len(self.game_state.players[0].hand[group])==0:
                for card_none in self.game_board.player1_widget.children:
                    if isinstance(card_none, CardWidget) and card_none.card is None and card_none.group == group:
                        self.game_board.player1_widget.remove_widget(card_none)
                        break            

            if self.win_cards: # win the cards
                self.game_board.floor_widget.rearrange_card_widgets(card_widget, callback=self.show_winning_cards)
            else:
                self.game_board.floor_widget.rearrange_card_widgets(card_widget, callback=self.flip_card_player1)
                self.click_flag = True  # Allow further clicks after the action is complete
            
    def flip_card_player1(self, *args):
        group = self.card_played1_group
        if self.game_state.players[0].hand[group]:
            card = self.game_state.players[0].hand[group][0]
            card_widget = CardWidget(card, name="Player1", group=group)
            self.game_board.player1_widget.add_widget(card_widget)

        self.click_flag = True  # Allow further clicks after the action is complete
        self.card_played1_group = None  # Reset the group after use
        self.win_cards, self.win_scores = [], []
        self.win_n = 0

        if self.soor:
            print("SOOOR occurred")
            # self.game_board.show_soor.update_scores(self.game_state.players[0].score, self.game_state.players[1].score)
            # self.soor = False

        if len(self.game_board.player2_widget.children) > 0:
            self.computer_move()
        else:
            print("No cards left for Player 1, waiting for next round")
            print(self.last_winner)
            Clock.schedule_once(lambda dt: self.finish_game(), 1)

    def show_winning_cards(self, card_widget):
        cards = self.win_cards[self.win_n]
        for card in cards:
            for idx, child in enumerate(self.game_board.floor_widget.children):
                if isinstance(child, CardWidget) and child.card == card:
                    child.pos = (child.pos[0], child.pos[1] + child.size[1] * 0.2)  # Move card up

    def on_touch_down_mat(self, mat, place):
        """Handle touch down events"""
        print(self.computer_flag)
        if self.computer_flag:
            cards = self.win_cards[self.win_n]
            self.move_card_to_mat(cards)
            self.flip_card_player2() 
            self.computer_flag = False
        else:           
            if len(self.win_cards) > 1:
                if (mat == "Mat1") and (place == "inside"):
                    cards = self.win_cards[self.win_n]
                    self.move_card_to_mat(cards)
                    self.flip_card_player1()
                else:
                    self.win_n += 1
                    if self.win_n >= len(self.win_cards):
                        self.win_n = 0
                    self.game_board.floor_widget.rearrange_card_widgets(None)
                    Clock.schedule_once(lambda dt: self.show_winning_cards(None), 0.5)
            elif len(self.win_cards) == 1:
                cards = self.win_cards[self.win_n]
                self.move_card_to_mat(cards)
                self.flip_card_player1()

    def move_card_to_mat(self, cards):
        for id, card in enumerate(cards):
            self.game_state.floor.remove_card(card)
            # Move winning card to mat1 
            for idx, child in enumerate(self.game_board.floor_widget.children):
                if isinstance(child, CardWidget) and child.card == card:
                    if self.computer_flag:
                        self.game_board.floor_widget.remove_widget(child)
                        self.game_board.mat2.add_widget(child)
                        anim = Animation(pos=(self.game_board.mat2.card_pos), duration=0.5)
                        if id == 0:
                            anim.bind(on_complete=lambda *args: self.game_board.mat2.draw_card())
                            anim.bind(on_complete=lambda *args: self.game_board.floor_widget.rearrange_card_widgets(None))
                        self.last_winner = "Computer"
                    else:
                        self.game_board.floor_widget.remove_widget(child)
                        self.game_board.mat1.add_widget(child)
                        anim = Animation(pos=(self.game_board.mat1.card_pos), duration=0.5)
                        if id==0:
                            anim.bind(on_complete=lambda *args: self.game_board.mat1.draw_card())
                            anim.bind(on_complete=lambda *args: self.game_board.floor_widget.rearrange_card_widgets(None))
                        self.last_winner = "Player1"
                    anim.start(child)

    def computer_move(self):
        """Handle computer's move"""
        
        self.computer_flag = True
        while True:
            best_move = random.randint(0,5)
            if len(self.game_state.players[1].hand[best_move]) > 0:
                break
        self.card_played2_group = best_move

        card = self.game_state.players[1].hand[best_move].pop(0)  # Remove the card from player's hand
        self.win_cards, self.win_scores, self.soor = self.game_state.floor.find_winning_cards(card)
        self.game_state.floor.add_card(card)  # Add card to the floor
        if len(self.win_scores)>0: #self.soor:
            self.player2_soor += 1
            print(f"Player 2 SOOOR count: {self.player2_soor}")
            self.game_board.show_soor.update_soors(self.player1_soor, self.player2_soor)

        for idx, child in enumerate(self.game_board.player2_widget.children):
            if isinstance(child, CardWidget) and child.card == card:
                self.game_board.player2_widget.remove_widget(child)
                self.game_board.floor_widget.add_widget(child)
                child.name = 'Floor'
                card_widget = child
                break

        # remove empty card widget from player2
        for card_none in self.game_board.player2_widget.children:
            if isinstance(card_none, CardWidget) and card_none.card is None and card_none.group == best_move:
                self.game_board.player2_widget.remove_widget(card_none)
                break

        if self.win_cards:
            self.game_board.floor_widget.rearrange_card_widgets(card_widget, callback=self.show_winning_cards)
        else:
            self.game_board.floor_widget.rearrange_card_widgets(card_widget, callback=self.flip_card_player2)
            self.computer_flag = False
        

    def flip_card_player2(self, *args):
        """Callback to add card to floor after animation"""
        print("Adding card to floor")
        group = self.card_played2_group
        print(f"Card group: {group}")
        if self.game_state.players[1].hand[group]:
            card = self.game_state.players[1].hand[group][0]
            card_widget = CardWidget(card, name="Computer", group=group)
            self.game_board.player2_widget.add_widget(card_widget)
            print(len(self.game_state.players[1].hand[group]))
            print(f"Card {card.rank} of {card.suit} is now in group {card_widget.name}")
        self.card_played2_group = None  # Reset the group after use
        self.win_cards, self.win_scores = [], []
        self.win_n = 0

        card_left1 = sum(len(hand) for hand in self.game_state.players[0].hand)
        card_left2 = sum(len(hand) for hand in self.game_state.players[1].hand)
        print(f"Cards left for Player 2: {card_left1} and Player 1: {card_left2}")
        if card_left1+card_left2 == 0:
            print("No cards left for Player 2, waiting for next round")
            Clock.schedule_once(lambda dt: self.finish_game(), 1)

    def finish_game(self):
        print("Finishing game")
        print(len(self.game_board.floor_widget.children))
        if len(self.game_board.floor_widget.children)==0:
            Clock.schedule_once(lambda dt: self.show_winning_cards_mat(), 1)
        else:
            for idx, child in enumerate(self.game_board.floor_widget.children[:]):
                if isinstance(child, CardWidget):
                    if self.last_winner == "Computer":
                        self.game_board.floor_widget.remove_widget(child)
                        self.game_state.floor.remove_card(child.card)
                        self.game_board.mat2.add_widget(child)
                        anim = Animation(pos=(self.game_board.mat2.card_pos), duration=0.5)
                        if idx == 0:
                            anim.bind(on_complete=lambda *args: Clock.schedule_once(lambda dt: self.show_winning_cards_mat(), 1))
                            anim.bind(on_complete=lambda *args: self.game_board.mat2.draw_card())
                    else:
                        self.game_board.floor_widget.remove_widget(child)
                        self.game_state.floor.remove_card(child.card)
                        self.game_board.mat1.add_widget(child)
                        anim = Animation(pos=(self.game_board.mat1.card_pos), duration=0.5)
                        if idx == 0:
                            anim.bind(on_complete=lambda *args: Clock.schedule_once(lambda dt: self.show_winning_cards_mat(), 1))
                            anim.bind(on_complete=lambda *args: self.game_board.mat1.draw_card())
                    anim.start(child)

    def show_winning_cards_mat(self):
        self.game_board.mat1.remove_card()
        self.game_board.mat2.remove_card()

        self.game_board.mat1.show_winning_cards()
        self.game_board.mat2.show_winning_cards()

    def count_clubs(self):
        print("Counting clubs")
        count1 = 0
        for child in self.game_board.mat1.children:
            if isinstance(child, CardWidget) and child.card.suit == "clubs":
                count1 += 1
        count2 = 0
        for child in self.game_board.mat2.children:
            if isinstance(child, CardWidget) and child.card.suit == "clubs":
                count2 += 1
        self.game_board.player1_score += count1
        self.game_board.player2_score += count2
        self.game_board.show_scores.update_scores(self.game_board.player1_score, self.game_board.player2_score)


    def count_positive(self):
        print("Counting positive scores")
        count1 = 0
        for child in self.game_board.mat1.children:
            if isinstance(child, CardWidget) and child.card.score > 0:
                count1 += child.card.score
        count2 = 0
        for child in self.game_board.mat2.children:
            if isinstance(child, CardWidget) and child.card.score > 0:
                count2 += child.card.score

        soor = self.player1_soor - self.player2_soor
        if soor > 0:
            count1 += soor * 10
        elif soor < 0:
            count2 += abs(soor) * 10

        self.game_board.player1_score += count1
        self.game_board.player2_score += count2
        self.game_board.show_scores.update_scores(self.game_board.player1_score, self.game_board.player2_score)

    def start_new_game(self):
        print("Starting new game")

        self.win_cards, self.win_scores = [], []
        self.win_n = 0
        self.click_flag = True  # Prevent multiple clicks
        self.card_played1_group = None
        self.card_played2_group = None
        self.computer_flag = False
        self.last_winner = None  # Track the last winner
        self.soor = False  # Track if a SOOOR has occurred

        self.game_board.difficulty_selection.player_start = not self.game_board.difficulty_selection.player_start

        self.game_state.setup()
        self.game_board.initialize_game_board()    


def print_hierarchy(widget, level=0):
    print("  " * level + f"{widget}")
    for child in widget.children:
        print_hierarchy(child, level + 1)