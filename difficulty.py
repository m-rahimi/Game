from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics import Rectangle, Color
from kivy.properties import StringProperty
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
import random

class DifficultySelection(FloatLayout):
    def __init__(self, callback, **kwargs):
        super(DifficultySelection, self).__init__(**kwargs)
        self.difficulty = "Not Selected"
        self.layout = FloatLayout()
        self.callback = callback

        # Create button container
        self.button_box = BoxLayout(
            orientation='vertical',
            spacing=10,
            size_hint=(0.5, 0.35),  # Smaller centered window
            pos_hint={'center_x': 0.5, 'center_y': 0.45 },
            padding=20,
        )
        
        # Add buttons
        self.easy_button = Button(text="Easy", background_color=(0.6, 0.8, 1, 1), font_size=40)
        self.easy_button.bind(on_press=self.difficulty_selected)

        self.medium_button = Button(text="Medium", background_color=(1, 0.6, 0.2, 1), font_size=40)
        self.medium_button.bind(on_press=self.difficulty_selected)

        self.hard_button = Button(text="Hard", background_color=(1, 0.3, 0.3, 1), font_size=40)
        self.hard_button.bind(on_press=self.difficulty_selected)
        
        self.button_box.add_widget(self.easy_button)
        self.button_box.add_widget(self.medium_button)
        self.button_box.add_widget(self.hard_button)
        self.layout.add_widget(self.button_box)
    
    def difficulty_selected(self, instance):
        if self.difficulty == "Not Selected":
            self.difficulty = instance.text
            self.max_depth = 0
            if self.difficulty == "Easy":
                self.max_depth = 2
            elif self.difficulty == "Medium":
                self.max_depth = 4
            elif self.difficulty == "Hard":
                self.max_depth = 6
                
            for key in self.button_box.children:
                if key != instance:
                    key.background_color[3] = key.background_color[3] - 0.2  # Dim the other buttons
                    key.text = ""

            x = random.randint(0, 1)
            if x == 0:
                self.player_start = True
                text = f"You start the game"
            else:
                self.player_start = False
                text = f"Computer starts the game"

            self.selection_label = Label(text=text,
                                    font_size=50,
#                                    size_hint=(0.5, 0.5),
                                    pos_hint={'center_x': 0.5, 'center_y': 0.65},
                                    color=(1, 1, 1, 1),
                                    font_name="Roboto-Bold.ttf")
            self.layout.add_widget(self.selection_label)
            Clock.schedule_once(lambda dt: self.callback(), 1)