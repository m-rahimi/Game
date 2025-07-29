from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Rectangle, Color
from kivy.core.window import Window

class CenteredRectangleWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Draw the rectangle
        with self.canvas:
            Color(1, 0, 0, 1)  # Red color
            self.rect = Rectangle()

        # Bind to window size changes
        Window.bind(size=self.on_window_resize)

        # Initialize the rectangle's position and size
        self.update_rectangle(Window.size)

    def on_window_resize(self, instance, size):
        """Callback when the window size changes."""
        self.update_rectangle(size)

    def update_rectangle(self, window_size):
        """Update the rectangle's size and position based on the window size."""
        window_width, window_height = window_size

        # Set the rectangle's size as a function of the window size
        rect_width = window_width * 0.5  # 50% of the window width
        rect_height = window_height * 0.3  # 30% of the window height

        # Center the rectangle in the window
        rect_x = (window_width - rect_width) / 2
        rect_y = (window_height - rect_height) / 2

        # Update the rectangle's position and size
        self.rect.pos = (rect_x, rect_y)
        self.rect.size = (rect_width, rect_height)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            print(f"touched at {touch.pos}")
            print(f"Widget pos: {self.pos}, size: {self.size}")
            return True
        return super().on_touch_down(touch)

class CenteredRectangleApp(App):
    def build(self):
        return CenteredRectangleWidget()

if __name__ == '__main__':
    CenteredRectangleApp().run()

# from kivy.app import App
# from kivy.uix.widget import Widget
# from kivy.uix.floatlayout import FloatLayout
# from kivy.graphics import Rectangle, Color
# from kivy.animation import Animation
# from kivy.clock import Clock

# class BoxWidget(Widget):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.size_hint = (None, None)
#         self.size = (100, 100)  # Box size
#         with self.canvas:
#             Color(1, 0, 0, 1)  # Red color
#             self.rect = Rectangle(pos=self.pos, size=self.size)
#         self.bind(pos=self.update_graphics, size=self.update_graphics)

#     def update_graphics(self, *args):
#         self.rect.pos = self.pos
#         self.rect.size = self.size

# class MainBoard(FloatLayout):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # Start drawing the first box, and pass a callback for when it's done
#         self.draw_first_box(callback=self.draw_second_box)

#     def draw_first_box(self, callback=None):
#         """Draw the first box and animate it."""
#         box = BoxWidget(pos=(100, 100))  # Initial position
#         self.add_widget(box)

#         # Animate the box to a new location
#         anim = Animation(pos=(300, 300), duration=3)
        
#         # If a callback is provided, bind it to the animation complete event
#         if callback:
#             anim.bind(on_complete=lambda *args: callback(2))
            
#         anim.start(box)
#         return box

#     def draw_second_box(self, n):
#         """Draw the second box after the animation finishes."""
#         box = BoxWidget(pos=(400, 400))  # Position for the second box
#         self.add_widget(box)
#         print(f"Second box drawn after first box animation completed with n={n}")

# class BoxApp(App):
#     def build(self):
#         return MainBoard()

# if __name__ == '__main__':
#     BoxApp().run()

# class BoxWidget(Widget):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.size_hint = (None, None)
#         self.size = (100, 100)  # Box size
#         with self.canvas:
#             Color(1, 0, 0, 1)  # Red color
#             self.rect = Rectangle(pos=self.pos, size=self.size)
#         self.bind(pos=self.update_graphics, size=self.update_graphics)

#     def update_graphics(self, *args):
#         self.rect.pos = self.pos
#         self.rect.size = self.size

# class MainBoard(FloatLayout):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.draw_first_box()

#     def draw_first_box(self):
#         """Draw the first box and animate it."""
#         box = BoxWidget(pos=(100, 100))  # Initial position
#         self.add_widget(box)

#         # Animate the box to a new location
#         anim = Animation(pos=(300, 300), duration=1)
#         anim.bind(on_complete=lambda *args: self.draw_second_box())
#         anim.start(box)

#     def draw_second_box(self):
#         """Draw the second box after the animation finishes."""
#         box = BoxWidget(pos=(400, 400))  # Position for the second box
#         self.add_widget(box)

# class BoxApp(App):
#     def build(self):
#         return MainBoard()

# if __name__ == '__main__':
#     BoxApp().run()