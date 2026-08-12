import cv2
import mediapipe as mp

# Path to our downloaded MediaPipe model
MODEL_PATH = "gesture/models/hand_landmarker.task"

# Create MediaPipe Hand Landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Camera started!")
print("Show your hand to the camera.")
print("Press Q to quit.")

# Create hand detector
with HandLandmarker.create_from_options(options) as landmarker:

    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Mirror the webcam
        frame = cv2.flip(frame, 1)

        # Convert OpenCV BGR image to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert to MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Timestamp must increase for video mode
        timestamp_ms = frame_number * 33
        frame_number += 1

        # Detect hands
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # Check whether a hand was detected
        if result.hand_landmarks:

            for hand_landmarks in result.hand_landmarks:

                # Draw the 21 hand landmarks
                height, width, _ = frame.shape

                for landmark in hand_landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1
                    )

                # Draw lines connecting landmarks
                connections = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12),
                    (9, 13), (13, 14), (14, 15), (15, 16),
                    (13, 17), (17, 18), (18, 19), (19, 20),
                    (0, 17)
                ]

                for start, end in connections:

                    x1 = int(hand_landmarks[start].x * width)
                    y1 = int(hand_landmarks[start].y * height)

                    x2 = int(hand_landmarks[end].x * width)
                    y2 = int(hand_landmarks[end].y * height)

                    cv2.line(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

            cv2.putText(
                frame,
                "HAND DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "NO HAND DETECTED",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Show webcam
        cv2.imshow(
            "Circuit Vision - Hand Tracking",
            frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("Camera closed.")