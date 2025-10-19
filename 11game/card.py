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

class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.score = find_score(rank, suit)
        self.value = RANKS_TO_NUMBER.get(rank, 11)
        self.prob = 1