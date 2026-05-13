import cv2
import time
from ultralytics import YOLO
from datetime import datetime
import numpy as np

# ========================= LOAD YOLO FACE MODEL =========================
model = YOLO("models/yolov8-face.pt")

def draw_corner_box(frame, x1, y1, x2, y2,
                    color=(0, 255, 255),
                    thickness=2):

    line_length = 25

    #Top-left
    cv2.line(frame, (x1, y1),
             (x1 + line_length, y1),
             color, thickness)

    cv2.line(frame, (x1, y1),
             (x1, y1 + line_length),
             color, thickness)

    #Top-right
    cv2.line(frame, (x2, y1),
             (x2 - line_length, y1),
             color, thickness)

    cv2.line(frame, (x2, y1),
             (x2, y1 + line_length),
             color, thickness)

    #Bottom-left
    cv2.line(frame, (x1, y2),
             (x1 + line_length, y2),
             color, thickness)

    cv2.line(frame, (x1, y2),
             (x1, y2 - line_length),
             color, thickness)

    #Bottom-right
    cv2.line(frame, (x2, y2),
             (x2 - line_length, y2),
             color, thickness)

    cv2.line(frame, (x2, y2),
             (x2, y2 - line_length),
             color, thickness)

cap = cv2.VideoCapture(1) #Opens Webcam

mode = "pixel" #Default mode

prev_time = 0 #FPS var.

#Main Loop
while True:

    ret, frame = cap.read() #Read frame

    if not ret:
        break

    results = model(frame, verbose=False) #Run YOLO face detection

    #Process detections
    for result in results:

        boxes = result.boxes

        for box in boxes:

            confidence = float(box.conf[0]) #Confidence score

            #Bounding box coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            #Prevent boundary issues
            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            #Width and height
            w = x2 - x1
            h = y2 - y1

            region = frame[y1:y2, x1:x2] #Crop face region

            #Skip invalid regions
            if region.size == 0:
                continue

            # ========================= PIXEL MODE =========================
            if mode == "pixel":

                small = cv2.resize(
                    region,
                    (16, 16)
                )

                pixelated = cv2.resize(
                    small,
                    (w, h),
                    interpolation=cv2.INTER_NEAREST
                )

                frame[y1:y2, x1:x2] = pixelated

            # ========================= BLACKOUT MODE =========================
            elif mode == "black":

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    -1
                )

            # ========================= BLUR MODE =========================
            elif mode == "blur":

                blurred = cv2.GaussianBlur(
                    region,
                    (99, 99),
                    30
                )

                frame[y1:y2, x1:x2] = blurred

            draw_corner_box(
                frame,
                x1,
                y1,
                x2,
                y2
            )

            #Confidence Text
            cv2.putText(
                frame,
                f"{int(confidence * 100)}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    #FPS Calculation
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    # ========================= TIMESTAMP =========================
    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cv2.putText(
        frame,
        timestamp,
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    #FPS Text
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    #Mode Text 
    cv2.putText(
        frame,
        f"Mode: {mode}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    #Controls
    cv2.putText(
        frame,
        "P=Pixel | B=Blackout | U=Blur | Q=Quit",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    #Resizing Window Output
    frame = cv2.resize(
        frame,
        (1280, 720)
    )

    cv2.imshow(
        "YOLO Edge-AI Surveillance",
        frame
    )

    #Keyboard Controls
    key = cv2.waitKey(1) & 0xFF

    #Quit
    if key == ord('q'):
        break

    #Modes:
    elif key == ord('p'):
        mode = "pixel"

    elif key == ord('b'):
        mode = "black"

    elif key == ord('u'):
        mode = "blur"

#Clean-up
cap.release()
cv2.destroyAllWindows()

