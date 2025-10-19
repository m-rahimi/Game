from PIL import Image, ImageDraw
import random
import os

# Configuration
background_size = (1600, 1200)  # Size of the background image
card_size = (222, 323)  # Standard card size (resized for aesthetics)
num_cards = 30
card_images_folder = "cards"  # Folder containing individual card images (ex: '2H.png', 'AS.png', etc.)
output_file = "card_background.png"
rectangle_margin = 5  # Padding around card image

# Load all card image filenames
card_files = [f for f in os.listdir(card_images_folder) if f.endswith(('.png', '.jpg'))]

# Create a blank background
background = Image.new("RGB", background_size, color=(0, 128, 0))  # Green felt background

for _ in range(num_cards):
    card_file = random.choice(card_files)
    card_path = os.path.join(card_images_folder, card_file)
    card_image = Image.open(card_path).convert("RGBA").resize(card_size)

    # Create new image with background rectangle behind card
    rect_size = (card_size[0] + rectangle_margin * 2, card_size[1] + rectangle_margin * 2)
    combined = Image.new("RGBA", rect_size, (255, 255, 255, 255))  # White background rectangle
    draw = ImageDraw.Draw(combined)
    
    # # Optional: choose a random rectangle color
    # rect_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255), 255)
    # draw.rectangle([0, 0, rect_size[0], rect_size[1]], fill=rect_color)
    
    # Paste card image centered over rectangle
    combined.paste(card_image, (rectangle_margin, rectangle_margin), card_image)

    # Random rotation
    angle = random.randint(-50, 50)
    rotated = combined.rotate(angle, expand=True)

    # Random position
    max_x = background_size[0] #- rotated.width
    max_y = background_size[1] #- rotated.height
    # if max_x < 0 or max_y < 0:
    #     continue  # Skip if card won't fit
    position = (random.randint(-20, max_x), random.randint(-20, max_y))

    # Paste onto background
    background.paste(rotated, position, rotated)

# Save result
background.save(output_file)
print(f"Background image saved as {output_file}")