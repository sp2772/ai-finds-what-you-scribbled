#example usage:
from tensorflow.keras.models import load_model
IMG_SIZE = 128
DATA_DIR = "/mnt/d/MyEverything/PythonProjects/Recent_projects/cnn_analysis/Hand_Drawing/quickdraw_images"
model = load_model("checkpoints/step_90000.keras")
def predict_image(model, image_path):
    """Predict the class of a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    classes = sorted(os.listdir(DATA_DIR))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    #preprocess image like training data
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    #view the image
    plt.imshow(img.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()
    # Make prediction
    prediction = model.predict(img)
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    #show top 10 predictions
    top_indices = np.argsort(prediction[0])[::-1][:20]
    top_classes = [list(class_to_idx.keys())[i] for i in top_indices]
    top_probs = prediction[0][top_indices]
    print(f"Predicted class: {list(class_to_idx.keys())[predicted_class_idx]}")
    print("Top 10 predictions:")
    for cls, prob in zip(top_classes, top_probs):
        print(f"  {cls}: {prob:.4f}")
    return predicted_class_idx
# Example usage
image_path = "/mnt/d/MyEverything/PythonProjects/Recent_projects/cnn_analysis/Hand_Drawing/clock.jpg"
#predicted_class_idx = predict_image(model, image_path)



#create a scribble pad for the user to draw on and have AI guess what it is every second
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def create_scribble_pad(model, img_size=128):
    """Create a scribble pad for the user to draw on and have AI guess what it is every second."""
    #create a pencil and eraser button and have user draw on a canvas
    canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    drawing = False
    brush_color = (255, 255, 255)  # White color for drawing
    brush_size = 5  # Size of the brush
    eraser_color = (0, 0, 0)  # Black color for erasing
    eraser_size = 20  # Size of the eraser
    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(canvas, (x, y), brush_size, brush_color, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
        elif event == cv2.EVENT_RBUTTONUP:
            drawing = False
            cv2.circle(canvas, (x, y), eraser_size, eraser_color, -1)
    cv2.namedWindow("Scribble Pad")
    cv2.setMouseCallback("Scribble Pad", draw)
    print("Left click to draw, right click to erase. Press 'q' to quit.")
    # Main loop
    
    
    while True:
        # Show the canvas
        cv2.imshow("Scribble Pad", canvas)
        
        # Predict every second
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if cv2.getTickCount() % cv2.getTickFrequency() < 1:  # Every second
            img = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)  # Add channel dimension
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            prediction = model.predict(img)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            
            classes = sorted(os.listdir(DATA_DIR))
            print(f"Predicted class: {classes[predicted_class_idx]}")
    
    cv2.destroyAllWindows()
    
# Example usage
# create_scribble_pad(model, img_size=IMG_SIZE)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Uncomment the line below to run the scribble pad
create_scribble_pad(model, img_size=IMG_SIZE)
# Note: The above code assumes you have a trained model and the necessary directories set up.
# If you want to run the scribble pad, make sure to uncomment the last line. 