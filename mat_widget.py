from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Line
from kivy.core.window import Window
from kivy.animation import Animation
from kivy.clock import Clock

ratio = 323 / 222


class MatWidget(Widget):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.line_width = 1.2
        self.line_points = []

        self.calculate_initial_pos_size()

        with self.canvas:
            Color(0, 0, 0, 1)
            self.rect = Rectangle(pos=self.pos, size=self.size)

        self.bind(size=self._update_rect, pos=self._update_rect)

    def calculate_initial_pos_size(self, *args):

        width = Window.width * 0.12
        height = width * ratio
        if height > Window.height * 0.25:
            height = Window.height * 0.3
            width = height / ratio
        rect_x = self.x + (Window.width * 0.8)
        rect_y1 = self.y + (Window.height * 0.15)
        rect_y2 = self.y + (Window.height - rect_y1 - height)
        
        if self.name == 'mat1':
            self.size = (width, height)
            self.pos = (rect_x, rect_y1)
        elif self.name == 'mat2':
            self.size = (width, height)
            self.pos = (rect_x, rect_y2)

    def _update_rect(self, *args):
        self.calculate_initial_pos_size()
        print(f"update_graphics called. pos: {self.pos}, size: {self.size}")
        if self.name == 'mat1':
            self.rect.pos = self.pos
            self.rect.size = self.size

        elif self.name == 'mat2':
            self.rect.pos = self.pos
            self.rect.size = self.size


    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            print(f"{self.name} touched at {touch.pos}")
            print(f"Widget pos: {self.pos}, size: {self.size}")
            return True
        return super().on_touch_down(touch)
    