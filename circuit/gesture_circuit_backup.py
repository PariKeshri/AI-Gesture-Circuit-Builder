import cv2
import mediapipe as mp
import numpy as np

WIDTH = 1200
HEIGHT = 700

MODEL_PATH = "gesture/models/hand_landmarker.task"

# ============================================================
# MEDIAPIPE
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
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.7
)

# ============================================================
# COMPONENT MENU
# ============================================================

menu = [
    ("BATTERY", 900, 170, 1170, 250),
    ("RESISTOR", 900, 280, 1170, 360),
    ("LED", 900, 390, 1170, 470),
    ("CAPACITOR", 900, 500, 1170, 580)
]

# Components that have been placed
placed = []

# ============================================================
# FIND COMPONENT
# ============================================================

def find_component(x, y):

    for name, x1, y1, x2, y2 in menu:

        if x1 <= x <= x2 and y1 <= y <= y2:
            return name

    return None


# ============================================================
# DRAW COMPONENT
# ============================================================

def draw_component(board, name, x, y):

    # Component box
    cv2.rectangle(
        board,
        (x, y),
        (x + 180, y + 90),
        (245, 245, 245),
        -1
    )

    cv2.rectangle(
        board,
        (x, y),
        (x + 180, y + 90),
        (40, 40, 40),
        2
    )

    # Different colors
    if name == "BATTERY":
        color = (70, 70, 220)

    elif name == "RESISTOR":
        color = (40, 150, 220)

    elif name == "LED":
        color = (40, 190, 100)

    else:
        color = (200, 120, 40)

    # Component symbol
    cv2.circle(
        board,
        (x + 90, y + 40),
        25,
        color,
        3
    )

    # Name
    cv2.putText(
        board,
        name,
        (x + 35, y + 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        2
    )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")
    exit()

# ============================================================
# SMOOTHING
# ============================================================

smooth_x = None
smooth_y = None

SMOOTHING = 0.18

# ============================================================
# STABILITY
# ============================================================

last_component = None
stable_frames = 0

# We require approximately 1 second of stability
REQUIRED_FRAMES = 30

# Prevent repeated placement
cooldown = 0

frame_number = 0

# ============================================================
# MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, camera = cap.read()

        if not success:
            break

        camera = cv2.flip(camera, 1)

        camera_height, camera_width, _ = camera.shape

        # ----------------------------------------------------
        # BOARD
        # ----------------------------------------------------

        board = np.ones(
            (HEIGHT, WIDTH, 3),
            dtype=np.uint8
        ) * 255

        # ----------------------------------------------------
        # TITLE
        # ----------------------------------------------------

        cv2.putText(
            board,
            "CIRCUIT VISION",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (20, 20, 20),
            2
        )

        cv2.putText(
            board,
            "Gesture-Based Circuit Builder",
            (30, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 100, 100),
            1
        )

        # ----------------------------------------------------
        # WORKSPACE
        # ----------------------------------------------------

        cv2.rectangle(
            board,
            (30, 110),
            (850, 620),
            (200, 200, 200),
            2
        )

        cv2.putText(
            board,
            "CIRCUIT WORKSPACE",
            (50, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (100, 100, 100),
            1
        )

        # ----------------------------------------------------
        # COMPONENT PANEL
        # ----------------------------------------------------

        cv2.rectangle(
            board,
            (870, 110),
            (1180, 620),
            (240, 240, 240),
            -1
        )

        cv2.rectangle(
            board,
            (870, 110),
            (1180, 620),
            (150, 150, 150),
            2
        )

        cv2.putText(
            board,
            "COMPONENTS",
            (925, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2
        )

        # ====================================================
        # MEDIAPIPE
        # ====================================================

        rgb = cv2.cvtColor(
            camera,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = frame_number * 33

        frame_number += 1

        result = landmarker.detect_for_video(
            mp_image,
            timestamp
        )

        cursor_x = None
        cursor_y = None

        current_component = None

        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            index_tip = hand[8]

            # Target cursor
            target_x = index_tip.x * WIDTH
            target_y = index_tip.y * HEIGHT

            # Smooth cursor
            if smooth_x is None:

                smooth_x = target_x
                smooth_y = target_y

            else:

                smooth_x += (
                    SMOOTHING *
                    (target_x - smooth_x)
                )

                smooth_y += (
                    SMOOTHING *
                    (target_y - smooth_y)
                )

            cursor_x = int(smooth_x)
            cursor_y = int(smooth_y)

            # Find component
            current_component = find_component(
                cursor_x,
                cursor_y
            )

            # =================================================
            # STABILITY
            # =================================================

            if current_component == last_component:

                stable_frames += 1

            else:

                stable_frames = 0
                last_component = current_component

            # =================================================
            # PLACEMENT
            # =================================================

            if (
                current_component is not None
                and
                stable_frames >= REQUIRED_FRAMES
                and
                cooldown == 0
            ):

                # Find a free position
                number = len(placed)

                x = 100 + (number % 3) * 240
                y = 190 + (number // 3) * 140

                placed.append(
                    {
                        "name": current_component,
                        "x": x,
                        "y": y
                    }
                )

                print(
                    "PLACED:",
                    current_component
                )

                # Reset
                stable_frames = 0
                last_component = None

                # Prevent immediate duplicate
                cooldown = 40

            # ------------------------------------------------
            # LANDMARKS
            # ------------------------------------------------

            for point in hand:

                px = int(
                    point.x * camera_width
                )

                py = int(
                    point.y * camera_height
                )

                cv2.circle(
                    camera,
                    (px, py),
                    3,
                    (0, 255, 0),
                    -1
                )

            # Index finger
            cv2.circle(
                camera,
                (
                    int(index_tip.x * camera_width),
                    int(index_tip.y * camera_height)
                ),
                10,
                (0, 0, 255),
                3
            )

        else:

            smooth_x = None
            smooth_y = None

            last_component = None
            stable_frames = 0

        # ====================================================
        # COOLDOWN
        # ====================================================

        if cooldown > 0:
            cooldown -= 1

        # ====================================================
        # DRAW MENU
        # ====================================================

        for name, x1, y1, x2, y2 in menu:

            if name == current_component:

                color = (255, 245, 190)
                border = (0, 190, 255)
                thickness = 5

            else:

                color = (255, 255, 255)
                border = (60, 60, 60)
                thickness = 2

            cv2.rectangle(
                board,
                (x1, y1),
                (x2, y2),
                color,
                -1
            )

            cv2.rectangle(
                board,
                (x1, y1),
                (x2, y2),
                border,
                thickness
            )

            cv2.putText(
                board,
                name,
                (x1 + 65, y1 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                2
            )

        # ====================================================
        # DRAW PLACED COMPONENTS
        # ====================================================

        for component in placed:

            draw_component(
                board,
                component["name"],
                component["x"],
                component["y"]
            )

        # ====================================================
        # CURSOR
        # ====================================================

        if cursor_x is not None:

            cv2.circle(
                board,
                (cursor_x, cursor_y),
                14,
                (0, 0, 255),
                3
            )

            cv2.circle(
                board,
                (cursor_x, cursor_y),
                4,
                (0, 0, 255),
                -1
            )

        # ====================================================
        # STATUS
        # ====================================================

        if current_component is not None:

            seconds = stable_frames / 30

            message = (
                "Holding: "
                + current_component
                + "   "
                + str(round(seconds, 1))
                + " / 1.0 sec"
            )

        else:

            message = "Point at a component"

        cv2.rectangle(
            board,
            (30, 640),
            (1180, 690),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            board,
            message,
            (50, 672),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # ====================================================
        # INSTRUCTIONS
        # ====================================================

        cv2.putText(
            board,
            "1. Point at a component",
            (50, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 80, 80),
            1
        )

        cv2.putText(
            board,
            "2. Keep finger steady for 1 second",
            (50, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 80, 80),
            1
        )

        cv2.putText(
            board,
            "3. Component appears in workspace",
            (50, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 80, 80),
            1
        )

        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            "Circuit Vision - Circuit Board",
            board
        )

        cv2.imshow(
            "Circuit Vision - Camera",
            camera
        )

        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Circuit Vision closed.")