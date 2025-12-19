from pathlib import Path
from collections import defaultdict
from PIL import Image
import random

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

IMG_EXTS = {".jpg", ".jpeg", ".png"}

SAMPLE_SIZE = 1000

def analyze_image_sizes_sample():
    all_images = []

    for p in DATA_DIR.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            all_images.append(p)

    sample = random.sample(all_images, min(SAMPLE_SIZE, len(all_images)))

    sizes = []
    for img_path in sample:
        with Image.open(img_path) as img:
            sizes.append(img.size)

    widths, heights = zip(*sizes)
    print("TAMAÑOS (MUESTRA)")
    print(f"Ancho: min={min(widths)}, max={max(widths)}")
    print(f"Alto:  min={min(heights)}, max={max(heights)}")

def analyze_dataset():
    counts = defaultdict(int)
    total = 0

    for surface_dir in DATA_DIR.iterdir():
        if not surface_dir.is_dir():
            continue

        surface = surface_dir.name

        for label_dir in surface_dir.iterdir():
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            n_images = sum(
                1 for p in label_dir.iterdir()
                if p.suffix.lower() in IMG_EXTS
            )

            counts[(surface, label)] = n_images
            total += n_images

    print("\n📊 IMÁGENES POR CATEGORÍA")
    for (surface, label), count in counts.items():
        print(f"{surface:10s} | {label:12s}: {count}")

    print(f"\nTotal de imágenes: {total}")

if __name__ == "__main__":
    analyze_dataset()
    analyze_image_sizes_sample()
