import itertools

class Floor:
    def __init__(self):
        self.hand = []  # List of Card objects

    def add_card(self, card):  
        self.hand.append(card)

    def remove_card(self, card):
        if card in self.hand:
            self.hand.remove(card)
            return 
        
    def find_winning_cards(self, card):
        win_cards = []
        win_scores = []

        if card.rank == 'jack':
            combo_cards = [c for c in self.hand if c.value < 12]
            win_cards.append(list(combo_cards) + [card])
        elif card.rank == 'queen':
            combo_cards = [c for c in self.hand if c.rank == 'queen']
            win_cards = [[c, card] for c in combo_cards]
        elif card.rank == 'king':
            combo_cards = [c for c in self.hand if c.rank == 'king']
            win_cards = [[c, card] for c in combo_cards]
        else:
            # Check all combinations of floor cards (any number of cards)
            values = [c.value for c in self.hand]
            for r in range(1, len(self.hand)+1):
                for combo in itertools.combinations(zip(self.hand, values), r):
                    combo_cards, combo_values = zip(*combo)
                    if sum(combo_values) + card.value == 11:
                        win_cards.append(list(combo_cards) + [card])

        if win_cards:
            for cards in win_cards:
                scores = sum(c.score for c in cards)
                clubs = sum(0.2 for c in cards if c.suit == 'clubs')
                win_scores.append(scores + clubs)

        return win_cards, win_scores

