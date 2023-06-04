import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

current_directory = Path(__file__).resolve().parent

# Load the pre-trained CNN model
model = tf.keras.models.load_model(f'{current_directory}/model_train/model/trained_model.h5')

# Load the cascade classifier for hand detection
hand_cascade = cv2.CascadeClassifier(f'{current_directory}/xml/haarcascade_hand.xml')

# Define the list of cookware
cookware = ["cup", "glass", "plate", "spoon", "fork", "knife"]


def predict_cookware_image(frame, cook_image):
    img = cv2.imread(f'{current_directory}/imagen/{cook_image}.jpg')
    
    objects = hand_cascade.detectMultiScale(
        img, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in objects:
        roi = img[max(0, y - 50):min(y + h + 50, img.shape[0]), max(0, x - 50):min(x + w + 50, img.shape[1])]
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi.reshape(1, 48, 48, 3) / 255.0
        cook = cookware[np.argmax(model.predict(roi))]
        cv2.rectangle(img, (max(0, x - 50), max(0, y - 50)), (min(x + w + 50, img.shape[1]), min(y + h + 50, img.shape[0])), (255, 0, 0), 2)
        cv2.putText(img, cook, (max(0, x - 50), max(0, y - 50)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2, cv2.LINE_AA)
        
        break  # Break the loop after processing the first detected object

    return img



def predict_cook_realtime(frame):
    faces = hand_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        roi = frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi.reshape(1, 48, 48, 3) / 255.0
        cook = cookware[np.argmax(model.predict(roi))]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, cook, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2, cv2.LINE_AA)

    return frame


def detection_realtime(video=False):
    if video:
        cap = cv2.VideoCapture(f'{current_directory}/imagen/Cookware.mp4')
    else:
        cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        try:
            ret, frame = cap.read()

            # Apply cookware detection to the frame
            frame_cook = predict_cook_realtime(frame)

            # Display the resulting frame
            cv2.imshow('Cookware Recognition', frame_cook)

            # Exit on ESC key
            if cv2.waitKey(1) == 27:
                break
        # Add an exception when the video ends
        except:
            print("Video ended")
            break

    # Release the capture and destroy the windows
    cap.release()
    cv2.destroyAllWindows()


def menu(opcion):
    if opcion == 1:
        # Load the image file
        cook_dict = {1: 'cup', 2: 'glass', 3: 'plate', 4: 'spoon', 5: 'fork', 6: 'knife'}
        option = int(input("\nSeleccione una imagen: \n1. cup\n2. glass\n3. plate\n4. spoon\n5. fork\n6. knife\n\t--> "))
        img = cv2.imread(f'{current_directory}/imagen/{cook_dict[option]}.jpg')
        # Predict the option in the image
        img_cook = predict_cookware_image(img, cook_dict[option])

        # Display the image
        cv2.imshow('Cookware Recognition', img_cook)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif opcion == 2:
        opt = int(input("\n1.Video or 2.Webcam?\n\t--> "))
        if opt == 1:
            detection_realtime(video=True)
        else:
            detection_realtime()


opcion = int(input(f"\nSeleccione una opción:\n1. Detectar utensilios de imágenes\n2. Detectar utensilios en tiempo real.\n\t--> "))
menu(opcion)
