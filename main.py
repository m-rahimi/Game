from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from game_state import GameState
from player import Player
from floor import Floor
from floor_widget import FloorWidget
from player_widget import PlayerWidget
from game_controller import GameController
from backgroung_widget import Background
from mat_widget import MatWidget

"""game/
  logic/
    card.py
    player.py
    game_state.py
  ui/
    card_widget.py
    player_widget.py
    floor_widget.py
  controllers/
    game_controller.py
  main.py"""

def print_hierarchy(widget, level=0):
    print("  " * level + f"{widget}")
    for child in widget.children:
        print_hierarchy(child, level + 1)

class GameBoard(FloatLayout):
    def __init__(self, controller, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.player1_widget = None
        self.player2_widget = None
        self.floor_widget = None
        self.mat1 = None
        self.mat2 = None

    def setup(self):
        self.background = Background()
        self.add_widget(self.background)

        self.mat1 = MatWidget(name='mat1')
#        self.mat2 = MatWidget(name='mat2')
        self.add_widget(self.mat1)
 #       self.add_widget(self.mat2)

        self.player1_widget = PlayerWidget(self.controller.game_state.players[0], 'Player1', self.controller)
        self.player2_widget = PlayerWidget(self.controller.game_state.players[1], 'Computer', self.controller)

        self.add_widget(self.player1_widget)
        self.add_widget(self.player2_widget)

        self.floor_widget = FloorWidget(self.controller.game_state.floor, self.controller)
        self.add_widget(self.floor_widget)

    # def update_ui(self):
    #     print('update_ui')
    #     # for player_widget in self.player_widgets.values():
    #     #     player_widget.update_cards()
    #     # self.floor_widget.update_cards()

    # def on_card_clicked(self, card_widget):
    #     print(f"Card clicked 2: {card_widget.card}")
    

class CardGameApp(App):
    def build(self):

        # Create game logic
        player1 = Player("Player1")
        player2 = Player("Computer")
        floor = Floor()
        game_state = GameState(players=[player1, player2], floor=floor)
        game_state.setup()

        # Create UI
        game_board = GameBoard(None)
        controller = GameController(game_state, game_board)
        game_board.controller = controller
        
        # # Connect logic and UI with controller
        game_board.setup()
        
        print_hierarchy(game_board)
        # # Start the game
        #controller.start_game()
        
        return game_board

if __name__ == '__main__':
    CardGameApp().run()