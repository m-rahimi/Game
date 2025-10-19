from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.core.window import Window
from kivy.uix.image import Image
from kivy.graphics import Color, Rectangle, Line

ratio = 323 / 222

class ShowScores(Widget):
    def __init__(self, score1, score2, **kwargs):
        super().__init__(**kwargs)
        self.initial_pos()
        self.score1 = str(score1).zfill(2)
        self.score2 = str(score2).zfill(2)

        # Create labels with initial font size
        self.label1 = Label(
            text=self.score1,
            font_size=self.calculate_font_size(self.score_size[1]),  # Calculate font size
            pos=self.score1_pos,
            size_hint=(None, None),
            size=self.score_size,
            halign='center',
            valign="middle",
            color=(0, 0, 1, 1),
            font_name="Roboto-Bold.ttf"
        )
        self.label2 = Label(
            text=self.score2,
            font_size=self.calculate_font_size(self.score_size[1]),  # Calculate font size
            pos=self.score2_pos,
            size_hint=(None, None),
            size=self.score_size,
            halign='center',
            valign="middle",
            color=(0, 0, 1, 1),
            font_name="Roboto-Bold.ttf"
        )

        with self.canvas:
            Color(1, 1, 1, 1)
            self.rect1 = Rectangle(pos=self.score1_pos, size=self.score_size)
            self.rect2 = Rectangle(pos=self.score2_pos, size=self.score_size)

        self.add_widget(self.label1)
        self.add_widget(self.label2)

        Window.bind(size=self.on_window_resize)

    def calculate_font_size(self, label_height):
        """Calculate font size based on the height of the label."""
        return label_height * 1.  # Set font size to 70% of the label height

    def initial_pos(self):
        width = Window.width * 0.12
        height = width * ratio
        if height > Window.height * 0.25:
            height = Window.height * 0.3
            width = height / ratio
        rect_x = self.x + (Window.width * 0.8)
        rect_y1 = self.y + (Window.height * 0.15)
        rect_y2 = self.y + (Window.height - rect_y1 - height)

        self.score1_pos = (rect_x, rect_y1 - height / 4 - height / 10)
        self.score2_pos = (rect_x, rect_y2 + height + height / 10)
        self.score_size = (int(width), int(height / 4))

    def on_window_resize(self, *args):
        self.initial_pos()
        self.update_text_size()
        self.label1.pos = self.score1_pos
        self.label2.pos = self.score2_pos
        self.label1.size = self.score_size
        self.label2.size = self.score_size

        self.rect1.pos = self.score1_pos
        self.rect1.size = self.score_size

        self.rect2.pos = self.score2_pos
        self.rect2.size = self.score_size

        # Update font size dynamically based on the new label height
        self.label1.font_size = self.calculate_font_size(self.score_size[1])
        self.label2.font_size = self.calculate_font_size(self.score_size[1])

    def update_text_size(self, *args):
        """Ensure the text is centered within the label."""
        self.label1.text_size = self.label1.size
        self.label2.text_size = self.label2.size

    def update_scores(self, score1, score2):
        """Update the scores and refresh the labels."""
        self.score1 = str(score1).zfill(2)
        self.score2 = str(score2).zfill(2)
        self.label1.text = self.score1
        self.label2.text = self.score2

class ShowSoor(Widget):
    def __init__(self, soor1, soor2, **kwargs):
        super().__init__(**kwargs)
        self.soor1 = soor1
        self.soor2 = soor2
        self.initial_pos()
        filename = "soor.png"
        self.card_image1 = []
        self.card_image2 = [] 

        for i in range(self.soor1):
            card_image = Image(source=filename, size=self.soor_size, pos=(self.soor1_pos[0], self.soor1_pos[1] + self.soor_size[1]*i), allow_stretch=True, keep_ratio=True, mipmap=True)
            self.add_widget(card_image)
            self.card_image1.append(card_image)

        for i in range(self.soor2):
            card_image = Image(source=filename, size=self.soor_size, pos=(self.soor2_pos[0], self.soor2_pos[1] - self.soor_size[1]*i), allow_stretch=True, keep_ratio=True, mipmap=True)
            self.add_widget(card_image)
            self.card_image2.append(card_image)

        Window.bind(size=self.on_window_resize)

    def initial_pos(self):
        width = Window.width * 0.12
        height = width * ratio
        if height > Window.height * 0.25:
            height = Window.height * 0.3
            width = height / ratio
        rect_x = self.x + (Window.width * 0.8)
        rect_y1 = self.y + (Window.height * 0.15)
        rect_y2 = self.y + (Window.height - rect_y1 - height)

        self.soor1_pos = (rect_x + width * 1.1, rect_y1)
        self.soor_size = (int(width / 3), int(height / 3))
        self.soor2_pos = (rect_x + width * 1.1, rect_y2 + height - self.soor_size[1])

    def on_window_resize(self, *args):
        self.initial_pos()
        for i, card_image in enumerate(self.card_image1):
            card_image.pos = (self.soor1_pos[0], self.soor1_pos[1] + self.soor_size[1]*i)
            card_image.size = self.soor_size

        for i, card_image in enumerate(self.card_image2):
            card_image.pos = (self.soor2_pos[0], self.soor2_pos[1] - self.soor_size[1]*i)
            card_image.size = self.soor_size

    def update_soors(self, soor1, soor2):
        """Update the SOOR counts and refresh the images."""
        self.soor1 = soor1
        self.soor2 = soor2

        # Clear existing images
        for card_image in self.card_image1:
            self.remove_widget(card_image)
        for card_image in self.card_image2:
            self.remove_widget(card_image)

        # Recreate images for new SOOR counts
        self.card_image1.clear()
        self.card_image2.clear()

        for i in range(self.soor1):
            card_image = Image(source="soor.png", size=self.soor_size, pos=(self.soor1_pos[0], self.soor1_pos[1] + self.soor_size[1]*i), allow_stretch=True, keep_ratio=True, mipmap=True)
            self.add_widget(card_image)
            self.card_image1.append(card_image)

        for i in range(self.soor2):
            card_image = Image(source="soor.png", size=self.soor_size, pos=(self.soor2_pos[0], self.soor2_pos[1] - self.soor_size[1]*i), allow_stretch=True, keep_ratio=True, mipmap=True)
            self.add_widget(card_image)
            self.card_image2.append(card_image)