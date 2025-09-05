from kivy.animation import Animation
from card_widget import CardWidget
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.app import App
import copy
import random
import itertools

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
        print(self.click_flag, self.computer_flag)
        if self.click_flag and self.computer_flag is False:
#            print("Player 1 clicked a card")
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
            if self.soor:
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

        if len(self.game_board.player2_widget.children) > 0:
            self.computer_move()
        else:
#            print("No cards left for Player 1, waiting for next round")
            Clock.schedule_once(lambda dt: self.finish_game(), 1)

    def show_winning_cards(self, card_widget):
        print("Showing winning cards")
        print("Winning cards:", len(self.win_cards), self.win_n)
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
        best_move, best_winning = self.find_best_move()

        self.computer_flag = True
        # while True:
        #     best_move = random.randint(0,5)
        #     if len(self.game_state.players[1].hand[best_move]) > 0:
        #         break
        self.card_played2_group = best_move

        card = self.game_state.players[1].hand[best_move].pop(0)  # Remove the card from player's hand
        self.win_cards, self.win_scores, self.soor = self.game_state.floor.find_winning_cards(card)
        self.game_state.floor.add_card(card)  # Add card to the floor
        if self.soor:
            self.player2_soor += 1
            print(f"Player 2 SOOOR count: {self.player2_soor}")
            self.game_board.show_soor.update_soors(self.player1_soor, self.player2_soor)

        # select the winning card with best
        if len(self.win_cards) > 1:
            print(self.win_cards)
            print(best_winning)
            for cards in self.win_cards:
                match = []
                if len(cards) == len(best_winning):
                    for c1 in cards:
                        for c2 in best_winning:
                            if c1.rank == c2.rank and c1.suit == c2.suit:
                                match.append(True)
                                break
                    if len(match) == len(best_winning):
                        self.win_cards = [cards]
                        break

        for idx, child in enumerate(self.game_board.player2_widget.children):
            if isinstance(child, CardWidget) and child.card == card:
                self.game_board.player2_widget.remove_widget(child)
                self.game_board.floor_widget.add_widget(child)
                child.name = 'Floor'
                card_widget = child
                break
 
        # remove empty card widget from player2
        if len(self.game_state.players[1].hand[best_move])==0:
            for card_none in self.game_board.player2_widget.children:
                if isinstance(card_none, CardWidget) and card_none.card is None and card_none.group == best_move:
                    self.game_board.player2_widget.remove_widget(card_none)
                    break

        if self.win_cards:
            self.game_board.floor_widget.rearrange_card_widgets(card_widget, callback=self.show_winning_cards)
        else:
            self.game_board.floor_widget.rearrange_card_widgets(card_widget, callback=self.flip_card_player2)
            self.computer_flag = False

    def find_best_move(self):
        player1 = []
        player2 = []
        floor = [copy.deepcopy(c) for c in self.game_state.floor.hand]
        player1_unplayed = [0 for _ in range(6)]
        player2_unplayed = [0 for _ in range(6)]
        unplayed_cards = []

        for ig, group in enumerate(self.game_state.players[0].hand):
            if group:
                for id, card in enumerate(group):
                    if id==0:
                        player1.append(copy.deepcopy(card))
                    if id > 0:
                        unplayed_cards.append(copy.deepcopy(card))
#                        player1_unplayed[ig] += 1
            else:
                player1.append(None)
                player1_unplayed[ig] = 0

        for ig, group in enumerate(self.game_state.players[1].hand):
            if group:
                for id, card in enumerate(group):
                    if id==0:
                        player2.append(copy.deepcopy(card))
                    if id > 0:
                        unplayed_cards.append(copy.deepcopy(card))
#                        player2_unplayed[ig] += 1
            else:
                player2.append(None)
                player2_unplayed[ig] = 0

        max_depth = self.game_board.difficulty_selection.max_depth
        best, best_moves, best_winning = self.min_max(player1, player1_unplayed, player2, player2_unplayed, floor, unplayed_cards, player=2, depth=0, max_depth=max_depth, score1=0, score2=0)
        rnd = random.randint(0, len(best_moves) - 1)
        print(f"Best moves for computer: {best_moves} and selected move: {best_moves[rnd]}")
        return best_moves[rnd], best_winning[rnd]

    def min_max(self, player1, player1_unplayed, player2, player2_unplayed, floor, unplayed_cards, player, depth, max_depth, score1, score2):
        if depth == max_depth or (player==1 and all(card is None for card in player1)) or (player==2 and all(card is None for card in player2)):
            print(f"Evaluating: score1={score1}, score2={score2}")
            return score2 - score1  # The evaluation function

        if player == 2: # computer start the evaluation
            best = float('-inf')
            best_moves = []
            best_move = None
            best_winning = []
            for id, card in enumerate(player2):
                if card:
                    win_cards, win_scores = self.find_winning_cards(floor, card)
                #    print(f"{['*' for _ in range(depth)]}  Computer card {card.rank} of {card.suit} with winning scores {win_scores} and floor {len(floor)}")

                    # create new player and floor for the next iteration
                    if player2_unplayed[id] > 0:
                        new_player2_unplayed = player2_unplayed[:] # each iteration we need to reduce it
                        new_player2_unplayed[id] -= 1
                        new_player2 = player2[:]
                        replace_id = random.randint(0,len(unplayed_cards)-1)
                        new_card = copy.deepcopy(unplayed_cards[replace_id])
                        new_card.prob = 1.0/len(unplayed_cards)
                        new_player2[id] = new_card
                    #    print(f"{['*' for _ in range(depth)]}  Computer new card {new_card.rank} of {new_card.suit} with prob {new_card.prob}")
                        new_unplayed_cards = unplayed_cards[:]
                        new_unplayed_cards.pop(replace_id)
                    else:
                        new_player2 = player2[:]
                        new_player2[id] = None                        
                        new_player2_unplayed = player2_unplayed[:]
                        new_unplayed_cards = unplayed_cards[:]

                    if len(win_cards)==0:
                        new_floor = floor[:]
                        new_floor.append(card)
                        value = self.min_max(player1, player1_unplayed, new_player2, new_player2_unplayed, new_floor, new_unplayed_cards, player=1, depth=depth+1, max_depth=max_depth, score1=score1, score2=score2)
                        if value > best:
                            best = max(best, value)
                            best_moves = [id]
                            best_winning = [None]
                        elif value == best:
                            best_moves.append(id)
                            best_winning.append(None)
                    else:
                        for cards, score in zip(win_cards, win_scores):
                            new_floor = [x for x in floor if x not in cards]
                            value = self.min_max(player1, player1_unplayed, new_player2, new_player2_unplayed, new_floor, new_unplayed_cards, player=1, depth=depth+1, max_depth=max_depth, score1=score1, score2=score2+score)
                            if value > best:
                                best = max(best, value)
                                best_moves = [id]
                                best_winning = [cards]
                            elif value == best:
                                best_moves.append(id)
                                best_winning.append(cards)
            if depth == 0:
                print(f"Best moves for computer at depth {depth}: {best_moves}, {best}")
                return best, best_moves, best_winning  # Return the best move for player 1 at depth 0
            else:
                return best
    
        elif player==1:
            best = float('inf')
            for id, card in enumerate(player1):
                if card:
                    win_cards, win_scores = self.find_winning_cards(floor, card)
                #    print(f"{['*' for _ in range(depth)]} Player card {card.rank} of {card.suit} with winning scores {win_scores} and floor {len(floor)}")
                    # create new player and floor for the next iteration
                    if player1_unplayed[id] > 0:
                        new_player1_unplayed = player1_unplayed[:]
                        new_player1_unplayed[id] -= 1
                        new_player1 = player1[:]
                        replace_id = random.randint(0,len(unplayed_cards)-1)
                        new_card = copy.deepcopy(unplayed_cards[replace_id])
                        new_card.prob = 1.0/len(unplayed_cards)
                        new_player1[id] = new_card
                    #    print(f"{['*' for _ in range(depth)]}  Player new card {new_card.rank} of {new_card.suit} with prob {new_card.prob}")
                        new_unplayed_cards = unplayed_cards[:]
                        new_unplayed_cards.pop(replace_id)
                    else:
                        new_player1 = player1[:]
                        new_player1[id] = None
                        new_player1_unplayed = player1_unplayed[:]
                        new_unplayed_cards = unplayed_cards[:]

                    if len(win_cards)==0:
                        new_floor = floor[:]
                        new_floor.append(card)
                        value = self.min_max(new_player1, new_player1_unplayed, player2, player2_unplayed, new_floor, new_unplayed_cards, player=2, depth=depth+1, max_depth=max_depth, score1=score1, score2=score2)
                        best = min(best, value)
                    else:
                        for cards, score in zip(win_cards, win_scores):
                            new_floor = [x for x in floor if x not in cards]
                            value = self.min_max(new_player1, new_player1_unplayed, player2, player2_unplayed, new_floor, new_unplayed_cards, player=2, depth=depth+1, max_depth=max_depth, score1=score1+score, score2=score2)
                            best = min(best, value) 

            return best

    def find_winning_cards(self, floor, card):
        win_cards = []
        win_scores = []

        if card.rank == 'jack':
            combo_cards = [c for c in floor if c.value < 12]
            win_cards.append(list(combo_cards) + [card])
        elif card.rank == 'queen':
            combo_cards = [c for c in floor if c.rank == 'queen']
            win_cards = [[c, card] for c in combo_cards]
        elif card.rank == 'king':
            combo_cards = [c for c in floor if c.rank == 'king']
            win_cards = [[c, card] for c in combo_cards]
        else:
            # Check all combinations of floor cards (any number of cards)
            values = [c.value for c in floor]
            for r in range(1, len(floor)+1):
                for combo in itertools.combinations(zip(floor, values), r):
                    combo_cards, combo_values = zip(*combo)
                    if sum(combo_values) + card.value == 11:
                        win_cards.append(list(combo_cards) + [card])

        value_cards = [c for c in floor if c.value <= 10]
        if card.rank == 'jack' and len(value_cards) == 0: #single jack is not a winning card
            win_cards = []

        if win_cards:
            for cards in win_cards:
                scores = sum(c.score for c in cards)
                clubs = sum(0.2 for c in cards if c.suit == 'clubs')
                win_scores.append((scores + clubs)*card.prob)

        if card.rank != 'jack' and len(win_cards) == 1 and len(win_cards[0]) == len(floor) + 1: #SOOOR
            print("SOOOR")
            win_scores[0] += 10*card.prob # Add 10 points for a single winning card

        return win_cards, win_scores

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

        winning_score = 52
        if self.game_board.player1_score >= winning_score or self.game_board.player2_score >= winning_score:
            print("Game over, resetting scores")
            app = App.get_running_app()
            app.end_game()
            self.game_board.player1_score = 0
            self.game_board.player2_score = 0
        else:
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


            self.game_board.difficulty_selection.player_start = not self.game_board.difficulty_selection.player_start

            self.game_state.setup()
            self.game_board.initialize_game_board()    


def print_hierarchy(widget, level=0):
    print("  " * level + f"{widget}")
    for child in widget.children:
        print_hierarchy(child, level + 1)