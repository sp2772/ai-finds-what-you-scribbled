# AI Finds What You Scribbled!

A prediction pipeline based on the QuickDraw Images public Dataset.

**FIND DJANGO IMPLEMENTATION OF THE MODEL IN [SCRIBBLE PROJECT](https://github.com/sp2772/Scribble_Project)**

## QuickDraw-Scribble-CNN

This repository provides a full pipeline for training, testing, and interacting with a CNN-based image classifier trained on the Google Quick, Draw! Dataset. It supports efficient parallel downloading, custom training strategies, and real-time drawing-based inference interfaces.

### About the Project

This project uses 338 classes (out of 345 available) from the Quick, Draw! dataset: [https://quickdraw.withgoogle.com/data](https://quickdraw.withgoogle.com/data) to train a CNN model that can recognize hand-drawn sketches in real time. The dataset consists of 255x255 grayscale drawings, which are resized to 128x128 and used for training a CNN. The project also includes multiple interfaces for inference: OpenCV-based and Tkinter-based real-time prediction GUIs.

### Repository Structure

1.  `Download_Dataset.py`
    * Downloads `.npy` files for each class directly from the QuickDraw dataset.
    * Saves them to a local directory, suitable for basic experimentation or training.

2.  `para2_Download.py`
    * Uses multithreading to speed up downloads from the QuickDraw dataset.
    * Each class downloads 100,000 images.
    * Utilizes `labels.txt` to determine which classes to download.

3.  `labels.txt`
    * A list of 345 available labels from the QuickDraw dataset.
    * Only the first 338 labels are used for training.

4.  `Hand_drawing.ipynb`
    * A basic notebook to test and train a CNN using the dataset downloaded into `quickdraw_images/`.
    * Demonstrates preprocessing, model building, training, and simple evaluation.

5.  `scribble_and_guess.py`
    * A minimal OpenCV-based GUI for users to draw on a canvas.
    * Every few seconds, the model predicts and displays the top classes for the current scribble.

6.  `interface.ipynb`
    * An advanced Tkinter interface allowing real-time model prediction on a drawing canvas.
    * Automatically fetches predictions every second and displays top-N predictions.
    * The drawing board is interactive, and prediction results are visualized live.

7.  `Training_epochs.ipynb`

---

### Summary of the Training Pipeline

* **Configuration and Imports**
    * Loads required libraries and sets global configs such as image size, dataset path, classes used, batch size, etc.

---

* **File Path & Label Extraction:**
    * Recursively reads all image paths and assigns class labels based on the folder names.

---

* **Progressive Data Partitioning:**
    * Implements a custom function `partition_class_images_progressively()` to simulate realistic training environments:
        * Breaks dataset into chunks (N, 2N, 3N,...mN number of mutually exclusive images per class taken at m'th epoch for fitting model progressively) across epochs.
    * Balances class distribution.
    * Ensures that each class contributes diverse samples progressively.

---

* **Model Definition:**
    * Builds a simple CNN using TensorFlow/Keras suited for quick sketch classification.

---

* **Training Logic:**
    * Loads dataset incrementally.
    * Normalizes and preprocesses images.
    * Trains in small chunks to simulate continuous data inflow.
    * Includes training checkpoint saving and early stopping.

---

* **Validation and Plotting:**
    * Tracks validation accuracy after each chunk.
    * Plots performance metrics and training trends.

---

* **Final Save:**
    * Model is saved after training.
    * Metadata such as class-index mappings are pickled for repeated use.

---

### How Training Works

* Dataset is downloaded using `para2_Download.py` or `Download_Dataset.py`.
* Training images are organized by class in folders.

---

### `Training_epochs.ipynb`

* Loads these images.
* Splits them into progressively increasing chunks to simulate time-based training.
* Trains the CNN incrementally using TensorFlow.

---

### The trained model is used by:

* `interface.ipynb` (Tkinter interface for predictions)
* or `scribble_and_guess.py` (OpenCV-based sketch predictor).
