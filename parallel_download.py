import os
from quickdraw import QuickDrawDataGroup
from concurrent.futures import ProcessPoolExecutor
from PIL import ImageFile

# Prevent truncated image errors
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_labels_from_file(filepath, num_labels):
    with open(filepath, 'r') as f:
        lines = f.read().splitlines()
    return lines[:num_labels]

def label_already_downloaded(label_dir, max_images):
    return os.path.exists(label_dir) and len([f for f in os.listdir(label_dir) if f.endswith('.png')]) >= max_images

def download_label_images(label, max_drawings, base_dir):
    label_dir = os.path.join(base_dir, label)
    if label_already_downloaded(label_dir, max_drawings):
        return f"✅ Skipped '{label}': already has {max_drawings} images."

    os.makedirs(label_dir, exist_ok=True)
    downloaded = 0
    try:
        qdg = QuickDrawDataGroup(label, max_drawings=max_drawings)
        drawings = list(qdg.drawings)

        for i, drawing in enumerate(drawings):
            image_path = os.path.join(label_dir, f"{label}_{i+1}.png")
            with open(image_path, 'wb') as f:
                drawing.image.save(f, format='PNG')
                f.flush()
                os.fsync(f.fileno())
            downloaded += 1

        return f"⬇️ Downloaded {downloaded} images for label '{label}'."
    except Exception as e:
        return f"❌ Failed for '{label}': {e}"

def download_and_save_parallel(labels, max_drawings_per_label=100, base_dir="quickdraw_images", num_workers=8):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print(f"\n🧵 Using {num_workers} parallel workers to download...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(download_label_images, label, max_drawings_per_label, base_dir)
            for label in labels
        ]
        for future in futures:
            print(future.result())

# === MAIN CONFIGURATION ===
LABEL_FILE = "labels.txt"             # Path to your label file
NUM_LABELS_TO_DOWNLOAD = 300           # Choose how many labels to use (e.g., first 50)
MAX_DRAWINGS_PER_LABEL = 100000       # Maximum drawings per label
NUM_PARALLEL_WORKERS = 9            # Number of CPU cores/threads to use (<= 16)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    labels_to_download = read_labels_from_file(LABEL_FILE, NUM_LABELS_TO_DOWNLOAD)
    download_and_save_parallel(
        labels_to_download,
        max_drawings_per_label=MAX_DRAWINGS_PER_LABEL,
        base_dir="quickdraw_images",
        num_workers=NUM_PARALLEL_WORKERS
    )

