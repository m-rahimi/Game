from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Line
from kivy.core.window import Window
from kivy.clock import Clock

ratio = 323 / 222

class Background(Widget):

    def __init__(self, **kwargs):
        super(Background, self).__init__(**kwargs)

        self.calculate_initial_pos_size_bg()

        with self.canvas:
            #25,124,84
            Color(25/255, 124/255, 84/255, 1)  # Green color
            self.bg = Rectangle(size=self.bg_size, pos=self.bg_pos)
            #79,108,36
            Color(79/255, 108/255, 36/255, 1)  # Green color
            self.rect1 = Rectangle(size=self.mat1_size, pos=self.mat1_pos)

            Color(79/255, 108/255, 36/255, 1)  # Green color
            self.rect2 = Rectangle(size=self.mat2_size, pos=self.mat2_pos)

        self.bind(size=self._update_rect, pos=self._update_rect)
        Clock.schedule_once(lambda dt: self._update_rect())

    def calculate_initial_pos_size_bg(self, *args):
        self.bg_pos = (0, 0)
        self.bg_size = (Window.width, Window.height)

        width = Window.width * 0.12
        height = width * ratio
        if height > Window.height * 0.25:
            height = Window.height * 0.3
            width = height / ratio
        rect_x = self.x + (Window.width * 0.8)
        rect_y1 = self.y + (Window.height * 0.15)
        rect_y2 = self.y + (Window.height - rect_y1 - height)
        
        # mat1
        self.mat1_size = (width, height)
        self.mat1_pos = (rect_x, rect_y1)

        # mat2
        self.mat2_size = (width, height)
        self.mat2_pos = (rect_x, rect_y2)  

    def _update_rect(self, *args):
        self.calculate_initial_pos_size_bg()

        self.bg.pos = self.bg_pos
        self.bg.size = self.bg_size

        self.rect1.size = self.mat1_size
        self.rect1.pos = self.mat1_pos
        self.rect2.size = self.mat2_size
        self.rect2.pos = self.mat2_pos