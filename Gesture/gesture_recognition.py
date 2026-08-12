import cv2
import mediapipe as mp
import math

# ============================================================
# CIRCUIT VISION - GESTURE RECOGNITION
# ============================================================

MODEL_PATH = "gesture/models/hand_landmarker.task"


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

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


# ============================================================
# DISTANCE FUNCTION
# ============================================================

def distance(p1, p2):
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


# ============================================================
# GESTURE RECOGNITION
# ============================================================

def recognize_gesture(hand):

    # --------------------------------------------------------
    # FINGER LANDMARKS
    # --------------------------------------------------------
    #
    # Thumb:
    # 2 = base
    # 3 = joint
    # 4 = tip
    #
    # Index:
    # 6 = joint
    # 8 = tip
    #
    # Middle:
    # 10 = joint
    # 12 = tip
    #
    # Ring:
    # 14 = joint
    # 16 = tip
    #
    # Pinky:
    # 18 = joint
    # 20 = tip
    #
    # Wrist:
    # 0
    # --------------------------------------------------------

    wrist = hand[0]

    # --------------------------------------------------------
    # OTHER FOUR FINGERS
    # --------------------------------------------------------

    index_up = hand[8].y < hand[6].y
    middle_up = hand[12].y < hand[10].y
    ring_up = hand[16].y < hand[14].y
    pinky_up = hand[20].y < hand[18].y

    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    thumb_tip = hand[4]
    thumb_joint = hand[3]

    # Thumb distance from wrist
    thumb_tip_distance = distance(thumb_tip, wrist)
    thumb_joint_distance = distance(thumb_joint, wrist)

    thumb_extended = (
        thumb_tip_distance >
        thumb_joint_distance * 1.12
    )

    # Thumb pointing upward
    thumb_up = (
        thumb_tip.y < thumb_joint.y - 0.02
        and
        thumb_tip.y < wrist.y
    )

    # ========================================================
    # 1. THUMBS UP → CONFIRM
    # ========================================================

    if (
        thumb_extended
        and thumb_up
        and not index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    ):
        return (
            "CONFIRM",
            "Thumbs up = Confirm"
        )

    # ========================================================
    # 2. INDEX FINGER → SELECT
    # ========================================================

    if (
        index_up
        and not middle_up
        and not ring_up
        and not pinky_up
    ):
        return (
            "SELECT",
            "Point at a component = Select"
        )

    # ========================================================
    # 3. TWO FINGERS → ROTATE
    # ========================================================

    if (
        index_up
        and middle_up
        and not ring_up
        and not pinky_up
    ):
        return (
            "ROTATE",
            "Two fingers = Rotate"
        )

    # ========================================================
    # 4. OPEN PALM → RELEASE
    # ========================================================

    if (
        index_up
        and middle_up
        and ring_up
        and pinky_up
    ):
        return (
            "RELEASE",
            "Open palm = Release"
        )

    # ========================================================
    # 5. FIST → GRAB / MOVE
    # ========================================================

    if (
        not index_up
        and not middle_up
        and not ring_up
        and not pinky_up
        and not thumb_extended
    ):
        return (
            "GRAB / MOVE",
            "Closed fist = Grab & Move"
        )

    # ========================================================
    # UNKNOWN
    # ========================================================

    return (
        "UNKNOWN",
        "Try one of the gestures below"
    )


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


# ============================================================
# START PROGRAM
# ============================================================

print()
print("==============================================")
print("             CIRCUIT VISION")
print("          GESTURE CONTROL SYSTEM")
print("==============================================")
print()
print("GESTURES:")
print()
print("☝  INDEX FINGER  → SELECT")
print("✊  FIST          → GRAB / MOVE")
print("✌  TWO FINGERS   → ROTATE")
print("✋  OPEN PALM     → RELEASE")
print("👍  THUMB UP      → CONFIRM")
print()
print("Press Q to exit.")
print()


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    frame_number = 0

    while True:

        # ----------------------------------------------------
        # READ CAMERA
        # ----------------------------------------------------

        success, frame = cap.read()

        if not success:

            print("Could not read camera.")

            break

        # Mirror image
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # ----------------------------------------------------
        # CONVERT IMAGE
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        timestamp_ms = frame_number * 33

        frame_number += 1

        # ----------------------------------------------------
        # DETECT HAND
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        gesture = "NO HAND"
        instruction = "Show your hand"


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            gesture, instruction = recognize_gesture(hand)

            # ------------------------------------------------
            # DRAW LANDMARKS
            # ------------------------------------------------

            for point in hand:

                x = int(point.x * width)
                y = int(point.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # ------------------------------------------------
            # HAND CONNECTIONS
            # ------------------------------------------------

            connections = [

                # Thumb
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                # Index
                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                # Middle
                (5, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                # Ring
                (9, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                # Pinky
                (13, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                # Palm
                (0, 17)
            ]

            for start, end in connections:

                x1 = int(hand[start].x * width)
                y1 = int(hand[start].y * height)

                x2 = int(hand[end].x * width)
                y2 = int(hand[end].y * height)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )


        # ====================================================
        # TOP INFORMATION PANEL
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 105),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "CIRCUIT VISION",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Gesture: " + gesture,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            instruction,
            (20, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 0),
            1
        )


        # ====================================================
        # GESTURE GUIDE
        # ====================================================

        guide_width = 285
        guide_height = 205

        guide_x = width - guide_width - 10
        guide_y = 120

        cv2.rectangle(
            frame,
            (guide_x, guide_y),
            (
                guide_x + guide_width,
                guide_y + guide_height
            ),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "GESTURE GUIDE",
            (guide_x + 15, guide_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        guide_text = [

            "Point     -> Select",

            "Fist      -> Grab / Move",

            "2 Fingers -> Rotate",

            "Open Palm -> Release",

            "Thumb Up  -> Confirm"
        ]

        y = guide_y + 62

        for text in guide_text:

            cv2.putText(
                frame,
                text,
                (guide_x + 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (255, 255, 255),
                1
            )

            y += 28


        # ====================================================
        # BOTTOM STATUS
        # ====================================================

        cv2.rectangle(
            frame,
            (0, height - 40),
            (width, height),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "Use hand gestures to control Circuit Vision",
            (20, height - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1
        )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "Circuit Vision - Gesture Control",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Circuit Vision closed.")