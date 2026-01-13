from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle

class Background(FloatLayout):
    def __init__(self, **kwargs):
        super(Background, self).__init__(**kwargs)

        # First: draw the solid background color in canvas.before
        with self.canvas.before:
            Color(25/255, 124/255, 84/255, 1)  # Green color
            self.background = Rectangle(size=self.size, pos=self.pos)
        
        # Second: draw the image in the regular canvas (on top of background)
        with self.canvas:
            # Make sure to set color to white (no tint) before drawing image
            Color(1, 1, 1, 1)  # White color (no tinting)
            self.image = Rectangle(source='card_background.png', size=self.size, pos=self.pos)

        # Bind both to update with layout changes
        self.bind(size=self._update_rect, pos=self._update_rect)

    def _update_rect(self, instance, value):
        """Update both background and image size and position."""
        self.background.size = instance.size
        self.background.pos = instance.pos
        self.image.size = instance.size
        self.image.pos = instance.pos

    def remove_image(self):
        """Remove the image from the canvas."""
        self.canvas.remove(self.image)
        print("Image removed after 10 seconds.")