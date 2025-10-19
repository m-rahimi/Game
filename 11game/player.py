class Player:
    def __init__(self, name):
        self.name = name
        self.hand = [[] for _ in range(6)]  # List of Card objects
        self.clubs = 0

    def add_card(self, card, group):
        self.hand[group].append(card)

    def remove_card(self, card):
        print(f"Removing card: {card.rank} of {card.suit} from player {self.name}")
        for group in self.hand:
            if card in group:
                group.remove(card)
                return