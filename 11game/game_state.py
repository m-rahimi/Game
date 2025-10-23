import random
from card import Card
from floor import Floor
from player import Player

SUITS = ["clubs", "diamonds", "hearts", "spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "jack", "queen", "king", "ace"]

class GameState:
    def __init__(self, players, floor):
        self.players = players
        self.floor = floor

    def setup(self):
        deck = [(rank, suit) for suit in SUITS for rank in RANKS]
        random.shuffle(deck)
        # Clear hands
        for player in self.players:
            for i in range(6):
                for j in range(4): # Assuming each player gets 1 card per group
                    if deck:
                        rank, suit = deck.pop()
                        card = Card(rank, suit)
                        player.add_card(card, i)

        for i in range(4):
            rank, suit = deck.pop()
            card = Card(rank, suit)
            self.floor.add_card(card)

    def find_best_move(self, max_depth, top_scores=2):
        print("Computer is making a move...")
        print(self.players[1].clubs)
        player1, player1_unplayed, player2, player2_unplayed, floor, unplayed_cards = self.create_tuple()

        zeros = [0 for _ in range(6)]
        scores = []
        for ig in range(6):
            if player2[ig] is not None:
                best, best_moves, best_winning = self.min_max_index(ig, player1, zeros, player2, zeros, floor, unplayed_cards, player=2, depth=0, max_depth=max_depth, score1=0, score2=0)

                if player2[ig][0] == 'jack' and best_winning[0] is None:
                    best = -20  # Slightly prefer not playing jack first
                    print("Jack penalty applied")
    
                scores.append(best)
                print(f"Best moves for computer at index {ig}: {best_moves}, {best}, {best_winning}")
            else:
                scores.append(float('-inf'))
        
        print(f"All scores: {scores}")
        top_scores = self.top_indices(scores, n=top_scores)
        print(f"Top scores so far: {top_scores}")

        scores = []
        results = []
        for ig in top_scores:
            best, best_moves, best_winning = self.min_max_index(ig, player1, player1_unplayed, player2, player2_unplayed, floor, unplayed_cards, player=2, depth=0, max_depth=4, score1=0, score2=0)

            if player2[ig][0] == 'jack' and best_winning[0] is None:
                best = -20  # Slightly prefer not playing jack first
                print("Jack penalty applied")
                
            scores.append(best)
            results.append((best_moves[0], best_winning[0]))  #TBD
            print(f"Best moves for computer at index {ig}: {best_moves}, {best}, {best_winning}")

        best_value = max(scores)
        indices = []
        for i, value in enumerate(scores):
            if value>best_value-0.1:
                indices.append(i)
        print(scores)
        print(indices)
        best_indices = indices[random.randint(0,len(indices)-1)] #scores.index(best_value)
        print(best_indices)
        return results[best_indices]


    def create_tuple(self):
        player1 = []
        player2 = []
        floor = [(c.rank, c.suit, c.score, c.value, c.prob) for c in self.floor.hand] # [copy.deepcopy(c) for c in self.game_state.floor.hand]
        player1_unplayed = [0 for _ in range(6)]
        player2_unplayed = [0 for _ in range(6)]
        unplayed_cards = []
        n_clubs = 0

        for ig, group in enumerate(self.players[0].hand):
            if group:
                for id, c in enumerate(group):
                    if id==0:
                        player1.append((c.rank, c.suit, c.score, c.value, c.prob))
                    if id > 0:
                        unplayed_cards.append((c.rank, c.suit, c.score, c.value, c.prob))
                        player1_unplayed[ig] += 1
            else:
                player1.append(None)
                player1_unplayed[ig] = 0

        for ig, group in enumerate(self.players[1].hand):
            if group:
                for id, c in enumerate(group):
                    if id==0:
                        player2.append((c.rank, c.suit, c.score, c.value, c.prob))
                    if id > 0:
                        unplayed_cards.append((c.rank, c.suit, c.score, c.value, c.prob))
                        player2_unplayed[ig] += 1
            else:
                player2.append(None)
                player2_unplayed[ig] = 0


        return player1, player1_unplayed, player2, player2_unplayed, floor, unplayed_cards
    
    def min_max_index(self, index, player1, player1_unplayed, player2, player2_unplayed, floor, unplayed_cards, player, depth, max_depth, score1, score2):
        if depth == max_depth or (player==1 and all(card is None for card in player1)) or (player==2 and all(card is None for card in player2)):
#            print(f"Evaluating: score1={score1}, score2={score2}")
            return score2 - score1  # The evaluation function

        if player == 2: # computer start the evaluation
            best = float('-inf')
            best_moves = []
            best_move = None
            best_winning = []
            for id, card in enumerate(player2):
                if (id!=index and depth==0 and index is not None):
                    continue
                else:
                    if card:
                        win_cards, win_scores = self.find_winning_cards(floor, card)

                        # create new player and floor for the next iteration
                        if player2_unplayed[id] > 0:
                            new_player2_unplayed = player2_unplayed[:] # each iteration we need to reduce it
                            new_player2_unplayed[id] -= 1
                            new_player2 = player2[:]
                            replace_id = random.randint(0,len(unplayed_cards)-1)
                            new_card = (unplayed_cards[replace_id][0], unplayed_cards[replace_id][1], unplayed_cards[replace_id][2], unplayed_cards[replace_id][3], 1.0/len(unplayed_cards))
                            new_player2[id] = new_card
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
                            value = self.min_max_index(None, player1, player1_unplayed, new_player2, new_player2_unplayed, new_floor, new_unplayed_cards, player=1, depth=depth+1, max_depth=max_depth, score1=score1, score2=score2)
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
                                value = self.min_max_index(None, player1, player1_unplayed, new_player2, new_player2_unplayed, new_floor, new_unplayed_cards, player=1, depth=depth+1, max_depth=max_depth, score1=score1, score2=score2+score)
                                if value > best:
                                    best = max(best, value)
                                    best_moves = [id]
                                    best_winning = [cards]
                                elif value == best:
                                    best_moves.append(id)
                                    best_winning.append(cards)
            if depth == 0:
#                print(f"Best moves for computer at depth {depth}: {best_moves}, {best}")
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
                        new_card = (unplayed_cards[replace_id][0], unplayed_cards[replace_id][1], unplayed_cards[replace_id][2], unplayed_cards[replace_id][3], 1.0/len(unplayed_cards))
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
                        value = self.min_max_index(None, new_player1, new_player1_unplayed, player2, player2_unplayed, new_floor, new_unplayed_cards, player=2, depth=depth+1, max_depth=max_depth, score1=score1, score2=score2)
                        best = min(best, value)
                    else:
                        for cards, score in zip(win_cards, win_scores):
                            new_floor = [x for x in floor if x not in cards]
                            value = self.min_max_index(None, new_player1, new_player1_unplayed, player2, player2_unplayed, new_floor, new_unplayed_cards, player=2, depth=depth+1, max_depth=max_depth, score1=score1+score, score2=score2)
                            best = min(best, value) 

            return best

    def find_combinations(self, floor, card, target=11):
        """Find all combinations of cards on the floor that sum to the target value."""
        win_cards = []
        win_scores = []

        def backtrack(start, current_combo, current_sum, current_score):
            # If the current sum + card.value equals the target, add the combination
            if current_sum + card[3] == target:
                win_cards.append(current_combo + [card])
                win_scores.append(current_score + card[2])  # Append the score of the card
                return

            # If the current sum exceeds the target, stop exploring this path
            if current_sum + card[3] > target:
                return

            # Explore further combinations
            for i in range(start, len(floor)):
                c = floor[i]
                backtrack(i + 1, current_combo + [c], current_sum + c[3], current_score + c[2])

        # Start backtracking
        backtrack(0, [], 0, 0)
        return win_cards

    def find_winning_cards(self, floor, card):
        win_cards = []
        win_scores = []

        if card[0] == 'jack':
            combo_cards = [c for c in floor if c[3] < 12]
            if len(combo_cards) > 0:
                win_cards.append(list(combo_cards) + [card])
        elif card[0] == 'queen':
            combo_cards = [c for c in floor if c[0] == 'queen']
            win_cards = [[c, card] for c in combo_cards]
        elif card[0] == 'king':
            combo_cards = [c for c in floor if c[0] == 'king']
            win_cards = [[c, card] for c in combo_cards]
        else:
            win_cards = self.find_combinations(floor, card)

        value_cards = [c for c in floor if c[3] < 12]
        if card[0] == 'jack' and len(value_cards) == 0: #single jack is not a winning card
            win_cards = []

        if self.players[1].clubs < 7:
            weight = self.players[1].clubs + 1
        else:
            weight = 0

        if win_cards:
            for cards in win_cards:
                scores = sum(c[2] for c in cards)
                clubs = sum(0.2*weight for c in cards if c[1] == 'clubs')
                win_scores.append((scores + clubs)*card[4])

        if card[0] != 'jack' and len(win_cards) == 1 and len(win_cards[0]) == len(floor) + 1: #SOOOR
            win_scores[0] += 10*card[4] # Add 10 points for a single winning card

        return win_cards, win_scores
    
    def top_indices(self, nums, n=2):
        indices = []
        for i in range(n):
            max_val = max(nums)
            max_idx = nums.index(max_val)
            if max_val == float('-inf'):
                break
            indices.append(max_idx)
            nums[max_idx] = float('-inf')  # mark this index so we don't pick
            while max_val == max(nums):
                max_val = max(nums)
                max_idx = nums.index(max_val)
                indices.append(max_idx)
                nums[max_idx] = float('-inf')  # mark this index so we don't pick
        return indices