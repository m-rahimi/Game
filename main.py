from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.clock import Clock
from player import Player
from floor import Floor
from floor_widget import FloorWidget
from player_widget import PlayerWidget
from game_controller import GameController
from backgroung_widget import Background, BackgroundImage
from difficulty import DifficultySelection
from mat_widget import MatWidget
from game_state import GameState
from card_widget import CardWidget
from deal_cards import deal_cards
from show_scores import ShowScores, ShowSoor


def print_hierarchy(widget, level=0):
    print("  " * level + f"{widget}")
    for child in widget.children:
        print_hierarchy(child, level + 1)

class GameBoard(FloatLayout):
    def __init__(self, controller, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.player1_score = 0
        self.player2_score = 0
        self.player1_widget = None
        self.player2_widget = None
        self.floor_widget = None
        self.mat1 = None
        self.mat2 = None

    def initialize_widgets(self):
        self.background = BackgroundImage()
        self.difficulty_selection = DifficultySelection(callback=self.initialize_game)
        self.background.add_widget(self.difficulty_selection.layout)
        self.add_widget(self.background)

    def initialize_game(self):
        print(f"Selected difficulty: {self.difficulty_selection.difficulty}")
        print(f"Player starts: {self.difficulty_selection.player_start}")
        self.background.remove_widget(self.difficulty_selection.layout)
        self.background.set_green_background()
        self.mat1 = MatWidget(name='Mat1')
        self.mat2 = MatWidget(name='Mat2')
        self.background.add_widget(self.mat1)
        self.background.add_widget(self.mat2)

        self.show_scores = ShowScores(self.player1_score, self.player2_score)
        self.background.add_widget(self.show_scores)

        self.show_soor = ShowSoor(0, 0)
        self.background.add_widget(self.show_soor)

        self.initialize_game_board()
        # self.deal_cards = deal_cards(self.controller.game_state.players[0], self.controller.game_state.players[1], self.controller.game_state.floor, self.initialize_game_board)
        # self.background.add_widget(self.deal_cards)

    def initialize_game_board(self):
        # self.background.remove_widget(self.deal_cards)

        self.player1_widget = PlayerWidget(self.controller.game_state.players[0], 'Player1', self.controller)
        self.player2_widget = PlayerWidget(self.controller.game_state.players[1], 'Computer', self.controller)

        self.background.add_widget(self.player1_widget)
        self.background.add_widget(self.player2_widget)

        self.floor_widget = FloorWidget(self.controller.game_state.floor, self.controller)
        self.background.add_widget(self.floor_widget)

        if not self.difficulty_selection.player_start:
            self.controller.computer_move()
        else:
            print("Player starts")

class CardGameApp(App):
    def initialize_game_logic(self, **kwargs):
        # Create game logic
        player1 = Player("Player1")
        player2 = Player("Computer")
        floor = Floor()
        self.game_state = GameState(players=[player1, player2], floor=floor)
        self.game_state.setup()

    def initialize_game_board(self):
        # Create UI
        self.game_board = GameBoard(None)
        self.controller = GameController(self.game_state, self.game_board)
        self.game_board.controller = self.controller

    def build(self):

        self.initialize_game_logic()
        self.initialize_game_board()

        self.game_board.initialize_widgets()

        # print_hierarchy(game_board)

        return self.game_board

if __name__ == '__main__':
    CardGameApp().run()

    # def on_touch_down(self, touch):
    #     """Handle mouse click events."""
    #     if not self.difficulty_selection.difficulty or self.difficulty_selection.difficulty == "Not Selected":
    #       print(f"Mouse clicked at: {touch.pos}")
    #       print(self.difficulty_selection.difficulty)
    #       Clock.schedule_once(self.background.set_green_background, 1)
    #       return super().on_touch_down(touch)
          

    # def on_touch_up(self, touch):
    #     if not self.difficulty_selection.difficulty or self.difficulty_selection.difficulty != "Not Selected":
    #       print(f"Mouse released at: {touch.pos}")
    #       print(self.difficulty_selection.difficulty)
    #     # self.mat2 = MatWidget(name='Mat2')
    #     # # self.mat1 = MatWidget(name='Mat1')
    #     # self.background.add_widget(self.mat2)
    #     # # self.add_widget(self.mat1)
    #     return super().on_touch_up(touch)