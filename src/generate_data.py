import os
import shutil
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

# Macros
WORDS = [
    "apple", "banana", "slayer", "perhaps", "restaurant", "tiger", "verdant", "machine", "titan", "shadow",
    "learning", "Gaming", "Explicit", "Krishak", "Byzantine", "Greek", "obey", "attitude", "yellow", "black",
    "ocean", "forest", "cyber", "dusk", "dawn", "spirit", "logic", "energy", "fusion", "vector",
    "omega", "pixel", "quantum", "neon", "vortex", "glitch", "matrix", "Diamond", "Azure", "Sapphire",
    "Emerald", "Ruby", "Topaz", "Aventurine", "golden", "copper", "iron", "steel", "whisper", "echo",
    "raven", "phoenix", "eclipse", "meteor", "zenith", "spectrum", "vivid", "luminous", "cosmic", "orbital",
    "gravity", "vital", "ignite", "spark", "flame", "ember", "blaze", "inferno", "horizon", "vista",
    "mirage", "solstice", "equinox", "cascade", "tide", "tempest", "breeze", "gale", "zephyr", "whirlwind",
    "cyclone", "storm", "thunder", "lightning", "drizzle", "monsoon", "frost", "glacier", "tundra", "meadow",
    "Aanan", "Kavin", "Akshat", "Divit", "Leeyun", "Ziggy", "Isabella", "Mia", "Elijah", "Olivia"
]
FONTS_EASY=["../data/fonts/arial.TTF"]
FONTS_HARD = ["../data/fonts/arial.TTF", "../data/fonts/Inkfree.TTF", "../data/fonts/BRITANIC.TTF", "../data/fonts/times.ttf", "../data/fonts/comic.ttf"]
OUTPUT_DIR = "../data/dataset"

IMAGE_SIZE = (180, 60)  # Width, Height
TRAIN_SPLIT = 0.8  # 80% train, 20% test

# Helper Functions

def apply_noise(image, mode='gaussian'):
    """Apply Gaussian or salt-and-pepper noise."""
    im_arr = np.asarray(image).astype(np.float32)

    if mode == 'gaussian':
        mean = 0
        std = 32
        noise = np.random.normal(mean, std, im_arr.shape)
        im_arr = im_arr + noise
    elif mode == 'sp':
        prob = 0.032
        salt_prob = prob / 2
        pepper_prob = prob / 2
        random_matrix = np.random.rand(*im_arr.shape[:2])
        im_arr[random_matrix < salt_prob] = 255
        im_arr[random_matrix > 1 - pepper_prob] = 0

    im_arr = np.clip(im_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(im_arr)

def random_capitalization(word):
    return "".join([char.upper() if random.random() > 0.6 else char.lower() for char in word])

def random_text_color():
    return tuple(np.random.randint(0, 128, size=3))

def apply_augmentation(image):
    angle = random.uniform(-5, 5)
    image = image.rotate(angle, expand=False, fillcolor="white")

    x_shift = random.randint(-5, 5)
    y_shift = random.randint(-5, 5)
    image = ImageOps.expand(image, border=(abs(x_shift), abs(y_shift)), fill="white")
    image = image.transform(
        IMAGE_SIZE,
        Image.AFFINE,
        (1, 0, -x_shift, 0, 1, -y_shift),
        fillcolor="white"
    )

    return image

def generate_captcha(word, font_path, output_dir, is_hard=False, augment=False):
    """Generate a single CAPTCHA image."""
    image = Image.new("RGB", IMAGE_SIZE, color="white")
    draw = ImageDraw.Draw(image)

    font_size = 28
    font = ImageFont.truetype(font_path, font_size)

    text_color = "black"
    if is_hard:
        word = random_capitalization(word)
        text_color = random_text_color()

    text_bbox = draw.textbbox((0, 0), word, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (IMAGE_SIZE[0] - text_width) / 2
    y = (IMAGE_SIZE[1] - text_height) / 4
    draw.text((x, y), word, fill=text_color, font=font)

    if augment:
      image = apply_augmentation(image)

    if is_hard:
        if random.random() > 0.5:
            image = apply_noise(image, mode='gaussian')
        else:
            image = apply_noise(image, mode='sp')


    filename = f"{word}_{hash(word + str(random.random()))}.png"
    image.save(os.path.join(output_dir, filename))
    return filename


# Generate Dataset
def generate_dataset(num_easy=1, num_hard=3, augment=False):
    # Set up directories
    base_dirs = {
        "easy_train": os.path.join(OUTPUT_DIR, "easy", "train"),
        "easy_test": os.path.join(OUTPUT_DIR, "easy", "test"),
        "hard_train": os.path.join(OUTPUT_DIR, "hard", "train"),
        "hard_test": os.path.join(OUTPUT_DIR, "hard", "test")
    }
    for base_dir in base_dirs.values():
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir, exist_ok=True)

    # Generate Easy and Hard Set
    for word in WORDS:
        # Easy Set
        easy_samples = [generate_captcha(word, random.choice(FONTS_EASY), OUTPUT_DIR, is_hard=False, augment=augment) for _ in range(num_easy)]
        easy_train = easy_samples[:int(TRAIN_SPLIT * num_easy)]
        easy_test = easy_samples[int(TRAIN_SPLIT * num_easy):]

        for img_path in easy_train:
            target_dir = os.path.join(base_dirs["easy_train"], word)
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(os.path.join(OUTPUT_DIR,img_path), target_dir)
        for img_path in easy_test:
            target_dir = os.path.join(base_dirs["easy_test"], word)
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(os.path.join(OUTPUT_DIR,img_path), target_dir)

        # Hard Set
        hard_samples = [generate_captcha(word, random.choice(FONTS_HARD), OUTPUT_DIR, is_hard=True, augment=augment) for _ in range(num_hard)]
        hard_train = hard_samples[:int(TRAIN_SPLIT * num_hard)]
        hard_test = hard_samples[int(TRAIN_SPLIT * num_hard):]

        for img_path in hard_train:
            target_dir = os.path.join(base_dirs["hard_train"], word)
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(os.path.join(OUTPUT_DIR,img_path), target_dir)
        for img_path in hard_test:
            target_dir = os.path.join(base_dirs["hard_test"], word)
            os.makedirs(target_dir, exist_ok=True)
            shutil.move(os.path.join(OUTPUT_DIR,img_path), target_dir)

