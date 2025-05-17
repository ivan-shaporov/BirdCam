import cv2
import numpy as np
import time
from datetime import datetime

# Try changing this index if the wrong camera is used (e.g., try 1, 2, etc.)
CAMERA_INDEX = 1  # 0 = front, 1 = rear (usually on Surface devices)

# Video writer settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20.0

# Motion detection parameters
PIXEL_DIFF_THRESHOLD = 50
MOTION_THRESHOLD = 0.1  # fraction of changed pixels
SHAKE_MOTION_THRESHOLD = 0.4  # ignore camera shake
NO_MOTION_TIMEOUT = 5  # seconds to wait before stopping recording

FRAME_AREA = FRAME_WIDTH * FRAME_HEIGHT


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Failed to open camera {CAMERA_INDEX}")
        return

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()

    recording = False
    last_motion_time = time.time()
    out = None

    print("Watching for motion... Press 'q' to quit.")

    while True:
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, PIXEL_DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        motion_area = sum(cv2.contourArea(c) for c in contours)
        affected_fraction = motion_area / FRAME_AREA
        motion_detected = MOTION_THRESHOLD < affected_fraction < SHAKE_MOTION_THRESHOLD

        if motion_detected:
            if not recording:
                file_name = f"../video/bird_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(affected_fraction * 100)}.avi"
                print(f"Motion detected — starting recording {file_name}.")
                out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'XVID'), FPS, (FRAME_WIDTH, FRAME_HEIGHT))
                recording = True
            last_motion_time = time.time()

        if recording:
            out.write(frame1)
            if time.time() - last_motion_time > NO_MOTION_TIMEOUT:
                print(f"No more motion — stopping recording {file_name}.")
                recording = False
                out.release()

        frame1 = frame2
        ret, frame2 = cap.read()

        if not ret:
            break

        # Optional preview window (can be removed in headless mode)
        cv2.imshow("BirdCam", frame1)
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
