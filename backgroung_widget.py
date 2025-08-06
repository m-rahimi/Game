from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Line
from kivy.core.window import Window
from kivy.uix.floatlayout import FloatLayout

class BackgroundImage(FloatLayout):
    def __init__(self, **kwargs):
        super(BackgroundImage, self).__init__(**kwargs)
        self.rect = None
        self.set_background_image()
        
    def set_background_image(self, source='card_background.png'):
        self.canvas.before.clear()
        with self.canvas.before:
            self.rect = Rectangle(source=source, 
                                 size=self.size, 
                                 pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
    
    def set_green_background(self):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(25/255, 124/255, 84/255, 1)  # Green color
            self.rect = Rectangle(size=self.size, 
                                pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
    
    def _update_rect(self, instance, value):
        if self.rect:
            self.rect.pos = instance.pos
            self.rect.size = instance.size

class Background(Widget):

    def __init__(self, **kwargs):
        super(Background, self).__init__(**kwargs)

        with self.canvas:
            Color(25/255, 124/255, 84/255, 1)  # Green color
            self.bg = Rectangle(size=self.bg_size, pos=self.bg_pos)

        self.bind(size=self._update_rect, pos=self._update_rect)
        # Clock.schedule_once(lambda dt: self._update_rect())

    def _update_rect(self, *args):
        self.bg.pos = self.bg_pos
        self.bg.size = self.bg_size