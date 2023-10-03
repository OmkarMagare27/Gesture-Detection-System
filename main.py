import cv2
import os
import time
import numpy as np
import tensorflow as tf
from tkinter import *
from PIL import Image, ImageTk

# 1. Data Collection
def capture_gestures():
    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        cv2.imshow('Gesture', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if not os.path.exists('gestures'):
                os.mkdir('gestures')
            cv2.imwrite(f'gestures/gesture_{time.time()}.png', frame)
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# 2. Data Preprocessing
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    # Implement your preprocessing steps here...
    return image

# 3. Model Development
def create_model():
    model = tf.keras.models.Sequential([
        # Define your model architecture here...
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 4. Gesture Recognition
def predict_gesture(model, image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(np.array([preprocessed_image]))
    gesture = np.argmax(prediction)
    return gesture

# 5. Action Mapping
def execute_action(gesture_id):
    actions = {0: action_0, 1: action_1, 2: action_2}
    if gesture_id in actions:
        actions[gesture_id]()

# 6. Example Action Functions
def action_0():
    print("Action 0 Triggered")

def action_1():
    print("Action 1 Triggered")

def action_2():
    print("Action 2 Triggered")

# 7. User Interface with Tkinter
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.video_source = 0  # use the first webcam on the system
        self.vid = cv2.VideoCapture(self.video_source)
        
        self.canvas = Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
        
        self.btn_snapshot = Button(window, text="Snapshot", width=10, command=self.snapshot)
        self.btn_snapshot.pack(anchor=CENTER, expand=True)
        
        self.update()
        self.window.mainloop()
        
    def snapshot(self):
        _, frame = self.vid.read()
        cv2.imwrite(f'gesture_{time.time()}.jpg', cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
    def update(self):
        _, frame = self.vid.read()
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.window.after(10, self.update)
        
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Example usage
# capture_gestures()
# model = create_model()
# model.fit(train_images, train_labels, epochs=5)
# gesture = predict_gesture(model, test_image)
# execute_action(gesture)

# GUI
window = Tk()
App(window, "Gesture Recognition")
