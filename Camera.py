import cv2
import time
import numpy as np

def main():
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame
        ret, frame = vid.read()

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Quit on press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the vid object
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

