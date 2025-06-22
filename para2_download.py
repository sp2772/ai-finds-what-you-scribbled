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

def get_existing_count(label_dir):
    if not os.path.exists(label_dir):
        return 0
    return len([f for f in os.listdir(label_dir) if f.endswith('.png')])

import gc

def download_label_images(label, max_drawings, base_dir):
    label_dir = os.path.join(base_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    existing_count = get_existing_count(label_dir)
    if existing_count >= max_drawings:
        return f"✅ Skipped '{label}': already has {existing_count} images."

    try:
        qdg = QuickDrawDataGroup(label, max_drawings=max_drawings)
        drawings = qdg.drawings  # generator object

        saved = 0
        for i, drawing in enumerate(drawings):
            if i < existing_count:
                continue
            if i >= max_drawings:
                break

            image_path = os.path.join(label_dir, f"{label}_{i+1}.png")
            with open(image_path, 'wb') as f:
                drawing.image.save(f, format='PNG')
                f.flush()
                os.fsync(f.fileno())
            saved += 1

        # ✅ Clear up memory manually
        del drawings
        del qdg
        gc.collect()

        return f"⬇️ Downloaded {saved} new images for label '{label}' (total: {existing_count + saved})."
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
LABEL_FILE = "newlabels.txt"
NUM_LABELS_TO_DOWNLOAD = 68
MAX_DRAWINGS_PER_LABEL = 100000
NUM_PARALLEL_WORKERS = 2

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
