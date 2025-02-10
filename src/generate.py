import os
import random
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Generate Corpus File
CORPUS_FILE = "corpus.txt"

# Load the corpus from the file
with open(CORPUS_FILE, "r") as f:
    WORDS = [line.strip() for line in f.readlines() if 4 <= len(line.strip()) <= 12]

FONTS_HARD = ["./arial.ttf", "./times.ttf", "./comic.ttf"]
OUTPUT_DIR = "./dataset"

IMAGE_SIZE = (180, 60)
TRAIN_SPLIT = 0.8

def random_capitalization(word):
    return "".join([char.upper() if random.random() > 0.6 else char.lower() for char in word])

def generate_captcha(word, font_path, output_dir, is_hard=False, augment=False):
    image = Image.new("RGB", IMAGE_SIZE, color="white")
    draw = ImageDraw.Draw(image)

    font_size = 28
    font = ImageFont.truetype(font_path, font_size)

    if not is_hard:
        text_color = "black" 
    else:
        text_color = tuple(np.random.randint(0, 128, size=3))
        word = random_capitalization(word)
    
    text_bbox = draw.textbbox((0, 0), word, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (IMAGE_SIZE[0] - text_width) / 2
    y = (IMAGE_SIZE[1] - text_height) / 4
    draw.text((x, y), word, fill=text_color, font=font)
    
    filename = f"{word}_{hash(word + str(random.random()))}.png"
    image.save(os.path.join(output_dir, filename))
    return filename

# Generate Dataset
def generate_dataset(num_easy=1, num_hard=3, augment=False):
    base_dirs = {
        "hard_train": os.path.join(OUTPUT_DIR, "hard", "train"),
        "hard_test": os.path.join(OUTPUT_DIR, "hard", "test")
    }
    for base_dir in base_dirs.values():
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)

    for word in WORDS:
        hard_samples = [generate_captcha(word, random.choice(FONTS_HARD), OUTPUT_DIR, is_hard=True, augment=augment) for _ in range(num_hard)]
        hard_train = hard_samples[:int(TRAIN_SPLIT * num_hard)]
        hard_test = hard_samples[int(TRAIN_SPLIT * num_hard):]

        for img_path in hard_train:
            target_dir = os.path.join(base_dirs["hard_train"], word)
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(os.path.join(OUTPUT_DIR, img_path), target_dir)
        for img_path in hard_test:
            target_dir = os.path.join(base_dirs["hard_test"], word)
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(os.path.join(OUTPUT_DIR, img_path), target_dir)

# Call dataset generation
generate_dataset(num_easy=1, num_hard=3, augment=False)