import cv2
import time
import numpy as np
import mediapipe as mp
import pickle
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

'''
To add a new gesture:
1. Set CURR_GESTURE to the name of the gesture
2. Press 'g' to capture the gesture, while you're holding the gesture up, repeat however many times you want
3. Press 'esc' to exit (this will save the gestures to a pickle file)

You can also delete saved_gestures.pickle to reset the saved gestures
'''

CURR_GESTURE = 'right_five'

GESTURES = {}

def normalize_gesture(landmarks):
    palm_center = landmarks[0].copy()
    for i in range(5, 18, 4):
        palm_center += landmarks[i]
    palm_center /= 5
    # Get direction from wrist to center of palm
    palm_vector = landmarks[0] - palm_center
    # Make the palm center the origin
    landmarks -= palm_center
    # Get angle of palm vector
    angle = np.arctan2(palm_vector[0], palm_vector[1])
    # Rotate the gesture to be vertical
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    landmarks = np.dot(landmarks, R.T)
    # Get the maximum distance from the palm center
    max_dist = max(np.max(np.abs(landmarks[:, 0])), np.max(np.abs(landmarks[:, 1])))
    # Normalize y and x
    landmarks[:, 0] /= max_dist
    landmarks[:, 1] /= max_dist
    return palm_center, max_dist, angle

def mse(a, b):
    return ((a - b)**2).mean(axis=None)

def recognize_gesture(landmarks):
    global GESTURES
    # Find the gesture with the minimum MSE
    min_mse = float('inf')
    min_gesture = 'none'
    for gesture, gesture_landmarks in GESTURES.items():
        curr_mse = mse(landmarks, gesture_landmarks)
        if curr_mse < min_mse:
            min_mse = curr_mse
            min_gesture = gesture
    return min_gesture

def main():
    global CURR_GESTURE, GESTURES
    # Try to load saved gestures from pickle file
    try:
        with open('saved_gestures.pickle', 'rb') as handle:
            GESTURES = pickle.load(handle)
    except:
        pass
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Flip image horizontally
            image = cv2.flip(image, 1)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            landmarks = None
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    # Convert landmarks to numpy array
                    landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
                    # Normalize the gesture
                    palm_center, max_dist, angle = normalize_gesture(landmarks)
                    # Recognize the gesture
                    found_gesture = recognize_gesture(landmarks)
                    # Draw the palm center
                    cv2.circle(image, (int(palm_center[0]*image.shape[1]), int(palm_center[1]*image.shape[0])), 10, (255, 0, 0), -1)
                    # Display text in top left corner
                    cv2.putText(image, found_gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', image)
            keypress = cv2.waitKey(1)
            if keypress == 27:
                with open('saved_gestures.pickle', 'wb') as handle:
                    pickle.dump(GESTURES, handle, protocol=pickle.HIGHEST_PROTOCOL)
                break
            # Check if 'g' is pressed
            elif keypress == ord('g') and landmarks is not None:
                # Add to saved gestures
                if CURR_GESTURE in GESTURES:
                    GESTURES[CURR_GESTURE].append(landmarks)
                else:
                    GESTURES[CURR_GESTURE] = [landmarks]
                print("Gesture captured")
    cap.release()

if __name__ == "__main__":
    main()

