from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Line
from kivy.core.window import Window
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.uix.image import Image
from card_widget import CardWidget

ratio = 323 / 222
CARD_FOLDER = "png"


class MatWidget(Widget):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.clubs_flag = False
        self.score_flag = False
        self.new_game = False

        # Prevent automatic resizing of the widget
        self.size_hint = (None, None)
        self.initial_size_pos()

        # Draw the rectangle
        with self.canvas:
            Color(79/255, 108/255, 36/255, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)

        # Bind to window size changes
        Window.bind(size=self.on_window_resize)

    def initial_size_pos(self):
        width = Window.width * 0.12
        height = width * ratio
        if height > Window.height * 0.25:
            height = Window.height * 0.3
            width = height / ratio
        rect_x = self.x + (Window.width * 0.8)
        rect_y1 = self.y + (Window.height * 0.15)
        rect_y2 = self.y + (Window.height - rect_y1 - height)
        # mat1
        if self.name == "Mat1":
            self.size = (int(width), int(height))
            self.pos = (rect_x, rect_y1)
        # mat2
        elif self.name == "Mat2":
            self.size = (int(width), int(height))
            self.pos = (rect_x, rect_y2)

        width = Window.width * 0.1
        height = width * ratio
        if height > Window.height * 0.25:
            height = Window.height * 0.25
            width = height / ratio
        self.card_size = (int(width), int(height))

        self.card_pos = (self.x + (self.width - self.card_size[0]) / 2, self.y + (self.height - self.card_size[1]) / 2)


    def on_window_resize(self, instance, size):
        window_width, window_height = Window.size

        width = window_width * 0.12
        height = width * ratio
        if height > Window.height * 0.25:
            height = Window.height * 0.3
            width = height / ratio
        rect_x = (window_width * 0.8)
        rect_y1 = (window_height * 0.15)
        rect_y2 = (window_height - rect_y1 - height)
        # mat1
        if self.name == "Mat1":
            self.rect.size = (width, height)
            self.rect.pos = (rect_x, rect_y1)
            print(f"Mat1 resized: {self.rect.size}, pos: {self.rect.pos}")
        # mat2
        elif self.name == "Mat2":
            self.rect.size = (width, height)
            self.rect.pos = (rect_x, rect_y2)

        width = Window.width * 0.1
        height = width * ratio
        if height > Window.height * 0.25:
            height = Window.height * 0.25
            width = height / ratio
        self.card_size = (int(width), int(height))

        self.card_pos = (self.x + (self.width - self.card_size[0]) / 2, self.y + (self.height - self.card_size[1]) / 2)


    def on_touch_down(self, touch):
        """Handle touch down events"""
        if self.clubs_flag:
            if self.name == "Mat1":
                self.parent.parent.controller.count_clubs()
            self.reset_clubs()
            Clock.schedule_once(lambda dt: self.shift_positive(), 1)
            return super().on_touch_down(touch)
        elif self.score_flag:
            print(f"Score flag is set for {self.name}")
            if self.name == "Mat1":
                self.parent.parent.controller.count_positive()
            self.score_flag = False
            self.new_game = True
            Clock.schedule_once(lambda dt: self.reset_clubs(), 1)
            return super().on_touch_down(touch)
        elif self.new_game:
            print(f"New game started from {self.name}")
            for child in self.children[:]:
                if isinstance(child, CardWidget):
                    self.remove_widget(child)
            if self.name == "Mat1":
                self.parent.parent.controller.start_new_game()
            self.new_game = False
            return super().on_touch_down(touch)
        else:
            if self.collide_point(*touch.pos):
                print(f"{self.name} touched inside")
                if hasattr(self.parent.parent.controller, 'on_touch_down_mat'):
                    if self.name == "Mat1":
                        self.parent.parent.controller.on_touch_down_mat(self.name, "inside")
    #            return True
            else:
                print(f"{self.name} touched outside")
                if hasattr(self.parent.parent.controller, 'on_touch_down_mat'):
                    if self.name == "Mat1":
                        self.parent.parent.controller.on_touch_down_mat(self.name, "outside")
    #            return True
            return super().on_touch_down(touch)
    
    def draw_card(self):
        self.remove_card()
        filename = "png/blue.png"
        self.card_image = Image(source=filename, size=self.card_size, pos=self.card_pos, allow_stretch=True, keep_ratio=True, mipmap=True)
        self.add_widget(self.card_image)

    def remove_card(self):
        if hasattr(self, 'card_image'):
            self.remove_widget(self.card_image)

    def show_winning_cards(self):
        card_count = sum(1 for child in self.children if isinstance(child, CardWidget))
        if card_count > 0:
            size = self.children[0].size if self.children else (100, 150)  # Default size if no cards
            card_y = self.children[0].pos[1] 
            card_x = Window.width * 0.05
            card_x_max = card_x + (size[0] + Window.width * 0.01) * 5.5
            if card_count <= 6:
                margin = Window.width * 0.01
            else:
                margin = (card_x_max - card_x) / (card_count-1) - size[0]
        
            # Update positions of all card widgets 
            for idx, child in enumerate(self.children):
                if isinstance(child, CardWidget):
                    x_pos = card_x + (size[0] + margin) * (card_count - idx - 1)
                    child.size = size
                    child.pos = (x_pos, card_y)

        Clock.schedule_once(lambda dt: self.shift_clubs(), 2)

    def shift_clubs(self):
        self.clubs_flag = True
        for idx, child in enumerate(self.children):
            if idx == 0:
                self.clubs_y_pos = child.pos[1]
            if isinstance(child, CardWidget) and child.card.suit == "clubs":
                child.pos = (child.pos[0], child.pos[1]+child.size[1] * 0.2)

    def reset_clubs(self):
        self.clubs_flag = False
        for idx, child in enumerate(self.children):
            if isinstance(child, CardWidget):
                child.pos = (child.pos[0], self.clubs_y_pos)

    def shift_positive(self):
        self.score_flag = True
        for idx, child in enumerate(self.children):
            if isinstance(child, CardWidget) and child.card.score > 0:
                child.pos = (child.pos[0], child.pos[1]+child.size[1] * 0.2)