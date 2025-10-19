import arcade
import random
import os
import itertools
import copy

# --- Constants ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CARD_WIDTH = 80
CARD_HEIGHT = 120
MARGIN = 10
mat_center_x = 680
mat_center_y1 = 500 - CARD_HEIGHT/2 - MARGIN
mat_center_y2 = 100 + CARD_HEIGHT/2 + MARGIN
mat_width = CARD_WIDTH + 2 * MARGIN
mat_height = CARD_HEIGHT + 2 * MARGIN

CARD_BACK_COLOR = arcade.color.GRAY
CARD_FOLDER = "png"  # Folder with card images like '2_of_clubs.png'
SUITS = ["clubs", "diamonds", "hearts", "spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace"]
RANKS_TO_NUMBER = {"2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "10": 10, "ace": 1, "jack": 11, "queen": 12, "king": 13}

def find_score(rank, suit):
    if suit == "clubs" and rank == "2":
        return 2
    elif suit == "diamonds" and rank == "10":
        return 3
    elif rank == "ace":
        return 1
    elif rank == "jack":
        return 1
    return 0

# --- Card Class ---
class Card:
    def __init__(self, rank, suit, owner):
        self.rank = rank
        self.suit = suit
        self.value = RANKS_TO_NUMBER.get(rank, 11)
        self.score = find_score(rank, suit) 
        self.owner = owner  # 'player1', 'player2', or 'floor'
        self.visible = False
        self.center_x = 0
        self.center_y = 0
        self.target_x = 0
        self.target_y = 0
        self.moving = False
        filename = f"{rank}_of_{suit}.png"
        self.texture = arcade.load_texture(os.path.join(CARD_FOLDER, filename))
        self.texture_back = arcade.load_texture(os.path.join(CARD_FOLDER, "blue.png"))

    def __deepcopy__(self, memo):
        # Create a new Card with the same basic attributes
        new_card = Card(self.rank, self.suit, self.owner)
        new_card.value = self.value
        new_card.score = self.score
        new_card.visible = self.visible
        new_card.center_x = self.center_x
        new_card.center_y = self.center_y
        new_card.target_x = self.target_x
        new_card.target_y = self.target_y
        new_card.moving = self.moving
        # Reuse the already loaded textures (do NOT deepcopy)
        new_card.texture = self.texture
        new_card.texture_back = self.texture_back
        return new_card


    def draw(self):
        # Draw white background
        arcade.draw_rectangle_filled(self.center_x, self.center_y, CARD_WIDTH, CARD_HEIGHT, arcade.color.WHITE)
        arcade.draw_rectangle_outline(self.center_x, self.center_y, CARD_WIDTH, CARD_HEIGHT, arcade.color.BLACK)

        scale_x = CARD_WIDTH / 222
        scale_y = CARD_HEIGHT / 323
        scale = min(scale_x, scale_y)

        if self.visible or self.moving:
            arcade.draw_scaled_texture_rectangle(self.center_x, self.center_y, self.texture, scale)
        elif (self.center_x == self.target_x and self.center_y == self.target_y):
            arcade.draw_scaled_texture_rectangle(self.center_x, self.center_y, self.texture_back, scale)

    def update_position(self):
        if self.moving:
            dx = self.target_x - self.center_x
            dy = self.target_y - self.center_y
            speed = 10  # pixels per frame

            distance = (dx ** 2 + dy ** 2) ** 0.5
            if distance < speed:
                self.center_x = self.target_x
                self.center_y = self.target_y
                self.moving = False
            else:
                self.center_x += speed * dx / distance
                self.center_y += speed * dy / distance

def calculate_scores(card, floor_cards, floor_values):
    collected_cards = []
    collected_cards_scores = []

    if card.rank == 'jack':
        combo_cards = [c for c in floor_cards if c.value < 12]
        collected_cards.append(list(combo_cards) + [card])
    elif card.rank == 'queen':
        combo_cards = [c for c in floor_cards if c.rank == 'queen']
        collected_cards = [[c, card] for c in combo_cards]
    elif card.rank == 'king':
        combo_cards = [c for c in floor_cards if c.rank == 'king']
        collected_cards = [[c, card] for c in combo_cards]
    else:
        # Check all combinations of floor cards (any number of cards)
        for r in range(1, len(floor_cards)+1):
            for combo in itertools.combinations(zip(floor_cards, floor_values), r):
                combo_cards, combo_values = zip(*combo)
                if sum(combo_values) + card.value == 11:
                    collected_cards.append(list(combo_cards) + [card])

    if collected_cards:
        for cards in collected_cards:          
            scores = sum(c.score for c in cards)
            clubs = sum(0.2 for c in cards if c.suit == 'clubs')
            collected_cards_scores.append(scores + clubs)

    return collected_cards, collected_cards_scores

def min_max(player1, player2, floor, player, depth, max_depth, score1, score2):
    if depth == max_depth or all(card is None for card in player1 + player2):
        return score1 - score2  # The evaluation function

    if player == 1:
        best = float('-inf')
        best_moves = []
        best_move = None
        for i, card in enumerate(player1):
            if card is None:
                continue
            collected_cards, collected_cards_scores = calculate_scores(card, floor, [x.value for x in floor])
            new_floor = copy.deepcopy(floor);  new_floor.append(card)  # new_floor = floor + [card]
            new_player1 = copy.deepcopy(player1)
            new_player1[i] = None
            if len(collected_cards) == 0:
                value = min_max(new_player1, player2, new_floor, 2, depth+1, max_depth, score1, score2)
            else:
                for score, cards in zip(collected_cards_scores, collected_cards):
                    new_floor = copy.deepcopy([x for x in floor if x not in cards])
                    value = min_max(new_player1, player2, new_floor, 2, depth+1, max_depth, score1 + score, score2)
            if value > best:
                best = max(best, value)
                best_moves = [i]
            elif value == best:
                best_moves.append(i)
        if depth == 0:
            print(f"Best moves for player 1 at depth {depth}: {best_moves}")
            best_move = random.choice(best_moves)
            return best, best_move  # Return the best move for player 1 at depth 0
        else:
            return best
        
    elif player == 2:
        best = float('inf')
        for i, card in enumerate(player2):
            if card is None:
                continue
            collected_cards, collected_cards_scores = calculate_scores(card, floor, [x.value for x in floor])
            new_floor = copy.deepcopy(floor);  new_floor.append(card)  # new_floor = floor + [card]
            new_player2 = copy.deepcopy(player2)
            new_player2[i] = None
            if len(collected_cards) == 0:
                value = min_max(player1, new_player2, new_floor, 1, depth+1, max_depth, score1, score2)
            else:
                for score, cards in zip(collected_cards_scores, collected_cards):
                    new_floor = copy.deepcopy([x for x in floor if x not in cards])
                    value = min_max(player1, new_player2, new_floor, 1, depth+1, max_depth, score1, score2 + score)
            best = min(best, value)

        return best

# --- Game Window ---
class CardGame(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Card Game with Smooth Movement")
        arcade.set_background_color(arcade.color.AMAZON)
        self.player1_cards = [[] for _ in range(6)]
        self.player2_cards = [[] for _ in range(6)]
        self.floor_cards = []
        self.collected_cards = []
        self.collected_cards_scores = []
        self.mat1 = []
        self.mat2 = []
        self.find_wining_cards_flag = False
        self.pending_computer_move = False
        self.pending_collect_cards = False
        self.score1 = 0
        self.score2 = 0
        self.index = -1
        self.count = 0
        self.last_winner = None
        self.move_sound = arcade.load_sound("sounds/card.wav")

    def setup(self):
        deck = [(rank, suit) for suit in SUITS for rank in RANKS]
#        random.seed(123)
        random.shuffle(deck)

        # Distribute 24 cards to each player
        for i in range(6):
            for _ in range(4):
                rank1, suit1 = deck.pop()
                card1 = Card(rank1, suit1, owner='computer')
                self.player1_cards[i].append(card1)

                rank2, suit2 = deck.pop()
                card2 = Card(rank2, suit2, owner='human')
                self.player2_cards[i].append(card2)

        # Place 4 cards on the floor
        for _ in range(4):
            rank, suit = deck.pop()
            card = Card(rank, suit, owner='floor')
            card.visible = True
            self.floor_cards.append(card)

        # Reveal top card of each group
        for i in range(6):
            self.player1_cards[i][0].visible = True
            self.player2_cards[i][0].visible = True

        self.set_card_positions()

    def set_card_positions(self):
        for index, group in enumerate(self.player1_cards):
            for i, card in enumerate(group):
                card.center_x = 70 + index * (CARD_WIDTH + MARGIN)
                card.center_y = 500 #- i * 5

        for index, group in enumerate(self.player2_cards):
            for i, card in enumerate(group):
                card.center_x = 70 + index * (CARD_WIDTH + MARGIN)
                card.center_y = 100 #+ i * 5

        for index, card in enumerate(self.floor_cards):
            card.center_x = 70 + index * (CARD_WIDTH + MARGIN)
            card.center_y = 300

    def move_card_to_floor(self, card, group):
        group.remove(card)
        self.floor_cards.append(card)

        # Play sound effect
        arcade.play_sound(self.move_sound)

        # Reveal next card in group
        if group:
            group[0].visible = True

        self.rearrange_floor_cards()

    
    def find_wining_cards(self):

        # Check for combinations on the floor that sum to 11 with the played card
        card = self.floor_cards[-1]
        floor_values = [c.value for c in self.floor_cards[:-1]]
        floor_cards = self.floor_cards[:-1].copy()
 
        self.collected_cards, self.collected_cards_scores = calculate_scores(card, floor_cards, floor_values)
        if self.collected_cards:
            self.pending_collect_cards = True

            max_score_index = self.collected_cards_scores.index(max(self.collected_cards_scores))
            print(f"Max score index: {max_score_index}, Scores: {self.collected_cards_scores}")

            for c in self.collected_cards[max_score_index]:
                c.target_y = 330
                c.moving = True

    def on_mouse_press(self, x, y, button, modifiers):
        max_score_index = 0

        if self.count >=0:
            print(f"Mouse pressed at ({x}, {y})")
            if self.pending_collect_cards or any(card.moving for card in self.floor_cards):
                print("Pending collect or move, ignoring mouse press.")
                card = self.floor_cards[-1]
                player = card.owner

                print(f"Pending collect for player: {player}")
                if player == 'human':
                    if len(self.collected_cards) == 1:
                        print("Collecting cards for human player 1")
                        self.pending_collect_cards = False
                        for c in self.collected_cards[0]:
                            c.target_x = mat_center_x
                            c.target_y = mat_center_y2
                            c.moving = True
                            c.visible = False
                            self.floor_cards.remove(c)
                            self.mat2.append(c)
                            self.last_winner = 'human'
                        self.rearrange_floor_cards()
                    else:
                        if abs(x - mat_center_x) <= mat_width / 2 and abs(y - mat_center_y2) <= mat_height / 2:
                            self.pending_collect_cards = False
                            print("Collecting cards for human player")
                            for c in self.collected_cards[self.index]:
                                c.target_x = mat_center_x
                                c.target_y = mat_center_y2
                                c.moving = True
                                c.visible = False
                                self.floor_cards.remove(c)
                                self.mat2.append(c)
                                self.last_winner = 'human'
                                self.index = -1  # Reset index for next collection
                            self.rearrange_floor_cards()

                        else:
                            if self.index == -1:
                                max_score_index = self.collected_cards_scores.index(max(self.collected_cards_scores))
                                self.index = max_score_index
                                for c in self.collected_cards[max_score_index]:
                                    c.target_y = 300
                                    c.moving = True

                            for c in self.collected_cards[self.index]:
                                c.target_y = 300
                                c.moving = True

                            self.index += 1
                            if self.index < len(self.collected_cards):
                                for c in self.collected_cards[self.index]:
                                    c.target_y = 330
                                    c.moving = True
                            else:
                                self.index = 0
                                for c in self.collected_cards[self.index]:
                                    c.target_y = 330
                                    c.moving = True
                            print(self.index, "Collecting cards for human player")

                elif player == 'computer':
                    print("Collecting cards for computer player")
                    self.pending_collect_cards = False
                    max_score_index = self.collected_cards_scores.index(max(self.collected_cards_scores))
                    for c in self.collected_cards[max_score_index]:
                        c.target_x = mat_center_x
                        c.target_y = mat_center_y1
                        c.moving = True
                        c.visible = False
                        self.floor_cards.remove(c)
                        self.mat1.append(c)
                        self.last_winner = 'computer'
                    self.rearrange_floor_cards()
                return
            
            if self.pending_computer_move or any(card.moving for card in self.floor_cards):
                return  # Ignore clicks while animations or moves pending
        
        if self.count == -1:
            nc = 0
            for card in self.mat1:
                if card.suit == 'clubs':
                    nc += 1
                    card.target_y += 30
                    card.moving = True
            if nc >= 7: self.score1 += 7

            nc = 0
            for card in self.mat2:
                if card.suit == 'clubs':
                    nc += 1
                    card.target_y += 30
                    card.moving = True
            if nc >= 7: self.score2 += 7
            self.count = -2

        elif self.count == -2:
            for card in self.mat2+self.mat1:
                if card.suit == 'clubs':
                    card.target_y -= 30
                    card.moving = True
            
            for card in self.mat1:
                if card.score >0:
                    self.score1 += card.score
                    card.target_y += 30
                    card.moving = True
            for card in self.mat2:
                if card.score >0:
                    self.score2 += card.score
                    card.target_y += 30
                    card.moving = True
            self.count = -3

                

        for i, group in enumerate(self.player2_cards):
            if group and group[0].visible:
                card = group[0]
                if abs(x - card.center_x) <= CARD_WIDTH / 2 and abs(y - card.center_y) <= CARD_HEIGHT / 2:
                    self.move_card_to_floor(card, group)
                    self.pending_computer_move = True
                    self.find_wining_cards_flag = True
                    break

    def computer_move(self):
        print("Computer is making a move...")
        player1 = [copy.deepcopy(x[0]) if x else None for x in self.player1_cards] #[x[0] if x else None for x in self.player1_cards]
        player2 = [copy.deepcopy(x[0]) if x else None for x in self.player2_cards] #[x[0] if x else None for x in self.player2_cards]
        floor = copy.deepcopy(self.floor_cards) #self.floor_cards[:] 
        best, id = min_max(player1, player2, floor, player=1, depth=0, max_depth=6, score1=0, score2=0)
        print(f"Computer move: {best}, id: {id}")
        if id is not None:
            card = self.player1_cards[id][0] 
        else:
            visible_cards = [(i, group[0]) for i, group in enumerate(self.player1_cards) if group and group[0].visible]
            if not visible_cards:
                return
            id, card = random.choice(visible_cards)
        self.move_card_to_floor(card, self.player1_cards[id])

    def on_update(self, delta_time):
        for card in self.floor_cards:
            card.update_position()

        for card in self.mat1 + self.mat2:
            card.update_position()

        if self.find_wining_cards_flag and not any(c.moving for c in self.floor_cards):
            print("Finding winning cards...")
            self.find_wining_cards()
            self.find_wining_cards_flag = False
            return

        # Wait for animation to finish before computer move
        if self.pending_computer_move and not self.find_wining_cards_flag and not self.pending_collect_cards and \
            not any(card.moving for card in self.floor_cards):
            self.computer_move()
            self.count += 1
            self.pending_computer_move = False
            self.find_wining_cards_flag = True
            for c in self.floor_cards:
                print(f"Card on last floor: {c.rank} of {c.suit}, score: {c.score}")
            return

        if self.count == 24 and not any(card.moving for card in self.floor_cards):
            self.count = -1
            print(self.last_winner, len(self.floor_cards))
            for c in self.floor_cards:
                print(f"Card on last floor: {c.rank} of {c.suit}, score: {c.score}")
                if self.last_winner == 'human':
                    print(f"Human player score: {self.score1}")
                    c.target_x = mat_center_x
                    c.target_y = mat_center_y2
                    c.moving = True
                    c.visible = False
#                    self.floor_cards.remove(c)
                    self.mat2.append(c)                    
                elif self.last_winner == 'computer':
                    print(f"Computer player score: {self.score2}")
                    c.target_x = mat_center_x
                    c.target_y = mat_center_y1
                    c.moving = True
                    c.visible = False
#                    self.floor_cards.remove(c)
                    self.mat1.append(c)

            self.show_wining_cards()
            print("Game over! Final scores:")

    def on_draw(self):
        arcade.start_render()

        # Draw mats
        mat_center_x = 680
        mat_center_y1 = 500 - CARD_HEIGHT/2 - MARGIN
        mat_center_y2 = 100 + CARD_HEIGHT/2 + MARGIN
        mat_width = CARD_WIDTH + 2 * MARGIN
        mat_height = CARD_HEIGHT + 2 * MARGIN

        arcade.draw_rectangle_filled(mat_center_x, mat_center_y1, mat_width, mat_height, arcade.color.DARK_OLIVE_GREEN)
        arcade.draw_rectangle_filled(mat_center_x, mat_center_y2, mat_width, mat_height, arcade.color.DARK_OLIVE_GREEN)

        # Draw scores
        arcade.draw_text(f"{self.score1:02d}", mat_center_x, mat_center_y1+CARD_HEIGHT/2+5*MARGIN, arcade.color.BLUE, 28, bold=True, 
                         width=mat_width,align="center", anchor_x="center", anchor_y="center")
        arcade.draw_text(f"{self.score2:02d}", mat_center_x, mat_center_y2-CARD_HEIGHT/2-5*MARGIN, arcade.color.BLUE, 28, bold=True, 
                         width=mat_width, align="center", anchor_x="center", anchor_y="center")

        # Draw player cards
        for group in self.player1_cards + self.player2_cards:
            for card in group:
                if card.visible:
                    card.draw()

        # Draw floor cards
        for card in self.floor_cards:
            card.draw()

        for card in self.mat1+self.mat2:
            card.draw()

    def rearrange_floor_cards(self):
        start_x = 70
        end_x = 70 + 5 * (CARD_WIDTH + MARGIN)
        count = len(self.floor_cards)

        if count <= 6:
            for index, card in enumerate(self.floor_cards):
                card.target_x = start_x + index * (CARD_WIDTH + MARGIN)
                card.target_y = 300
                card.moving = True

        else:
            # Distribute cards evenly across the available space
            step = (end_x - start_x) / (count - 1)
            for index, card in enumerate(self.floor_cards):
                card.target_x = start_x + index * step
                card.target_y = 300
                card.moving = True

    def show_wining_cards(self):
        start_x = 70
        end_x = 70 + 5 * (CARD_WIDTH + MARGIN)

        count1 = len(self.mat1)
        count2 = len(self.mat2)

        step1 = (end_x - start_x) / (count1 - 1)
        step2 = (end_x - start_x) / (count2 - 1)

        for index, card in enumerate(self.mat1):
            print(f"Card on mat1: {card.rank} of {card.suit}, score: {card.score}")
            card.target_x = start_x + index * step1
            card.target_y = 500
            card.visible = True
            card.moving = True

        for index, card in enumerate(self.mat2):
            print(f"Card on mat2: {card.rank} of {card.suit}, score: {card.score}")
            card.target_x = start_x + index * step2
            card.target_y = 100
            card.visible = True
            card.moving = True

# --- Run the Game ---
def main():
    game = CardGame()
    game.setup()
    arcade.run()

if __name__ == "__main__":
    main()
