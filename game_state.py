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
                for j in range(1): # Assuming each player gets 1 card per group
                    if deck:
                        rank, suit = deck.pop()
                        card = Card(rank, suit)
                        player.add_card(card, i)

        for i in range(4):
            rank, suit = deck.pop()
            card = Card(rank, suit)
            self.floor.add_card(card)