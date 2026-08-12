import cv2
import mediapipe as mp
import numpy as np
import math

# ============================================================
# CIRCUIT VISION
# Gesture Based Circuit Builder
# ============================================================

WIDTH = 1200
HEIGHT = 700

# IMPORTANT:
# Your models folder is inside Gesture, not circuit.
MODEL_PATH = "Gesture/models/hand_landmarker.task"


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


# ============================================================
# PLACED COMPONENTS
# ============================================================

placed = []


# ============================================================
# FIND COMPONENT IN MENU
# ============================================================

def find_component(x, y):

    for name, x1, y1, x2, y2 in menu:

        if x1 <= x <= x2 and y1 <= y <= y2:
            return name

    return None


# ============================================================
# CHECK WHETHER CURSOR IS ON PLACED COMPONENT
# ============================================================

def find_placed_component(x, y):

    # Search from last component to first
    # so the top component is selected first.

    for component in reversed(placed):

        cx = component["x"]
        cy = component["y"]

        if (
            cx <= x <= cx + 180
            and
            cy <= y <= cy + 90
        ):
            return component

    return None


# ============================================================
# DRAW COMPONENT
# ============================================================

def draw_component(board, name, x, y, selected=False):

    # --------------------------------------------------------
    # Component background
    # --------------------------------------------------------

    if selected:
        background = (210, 245, 255)
        border = (0, 150, 255)
        thickness = 4
    else:
        background = (245, 245, 245)
        border = (50, 50, 50)
        thickness = 2

    cv2.rectangle(
        board,
        (x, y),
        (x + 180, y + 90),
        background,
        -1
    )

    cv2.rectangle(
        board,
        (x, y),
        (x + 180, y + 90),
        border,
        thickness
    )

    # --------------------------------------------------------
    # Component colors
    # --------------------------------------------------------

    if name == "BATTERY":

        color = (70, 70, 220)

        # Battery symbol
        cv2.line(
            board,
            (x + 50, y + 30),
            (x + 50, y + 60),
            color,
            5
        )

        cv2.line(
            board,
            (x + 75, y + 20),
            (x + 75, y + 70),
            color,
            5
        )

        cv2.line(
            board,
            (x + 30, y + 45),
            (x + 50, y + 45),
            color,
            3
        )

        cv2.line(
            board,
            (x + 75, y + 45),
            (x + 100, y + 45),
            color,
            3
        )

    elif name == "RESISTOR":

        color = (40, 130, 220)

        points = np.array(
            [
                [x + 30, y + 45],
                [x + 50, y + 25],
                [x + 70, y + 65],
                [x + 90, y + 25],
                [x + 110, y + 65],
                [x + 130, y + 45]
            ],
            np.int32
        )

        cv2.polylines(
            board,
            [points],
            False,
            color,
            4
        )

    elif name == "LED":

        color = (40, 180, 90)

        cv2.circle(
            board,
            (x + 75, y + 45),
            25,
            color,
            3
        )

        cv2.line(
            board,
            (x + 25, y + 45),
            (x + 50, y + 45),
            color,
            3
        )

        cv2.line(
            board,
            (x + 100, y + 45),
            (x + 130, y + 45),
            color,
            3
        )

    else:

        color = (200, 120, 40)

        cv2.line(
            board,
            (x + 65, y + 20),
            (x + 65, y + 70),
            color,
            5
        )

        cv2.line(
            board,
            (x + 90, y + 20),
            (x + 90, y + 70),
            color,
            5
        )

        cv2.line(
            board,
            (x + 30, y + 45),
            (x + 65, y + 45),
            color,
            3
        )

        cv2.line(
            board,
            (x + 90, y + 45),
            (x + 125, y + 45),
            color,
            3
        )

    # --------------------------------------------------------
    # Component name
    # --------------------------------------------------------

    cv2.putText(
        board,
        name,
        (x + 25, y + 82),
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

    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# CURSOR SMOOTHING
# ============================================================

smooth_x = None
smooth_y = None

SMOOTHING = 0.20


# ============================================================
# PLACEMENT STABILITY
# ============================================================

last_menu_component = None
stable_frames = 0

REQUIRED_FRAMES = 30


# ============================================================
# PINCH / DRAGGING
# ============================================================

dragging = False
dragged_component = None

grab_offset_x = 0
grab_offset_y = 0

# Smaller = harder to pinch
# Larger = easier to pinch
PINCH_THRESHOLD = 0.055


# ============================================================
# COOLDOWN
# ============================================================

cooldown = 0


# ============================================================
# FRAME NUMBER
# ============================================================

frame_number = 0


# ============================================================
# MEDIAPIPE LOOP
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        success, camera = cap.read()

        if not success:
            print("Could not read webcam.")
            break

        # Mirror camera
        camera = cv2.flip(camera, 1)

        camera_height, camera_width, _ = camera.shape


        # ====================================================
        # CREATE WHITE BOARD
        # ====================================================

        board = np.ones(
            (HEIGHT, WIDTH, 3),
            dtype=np.uint8
        ) * 255


        # ====================================================
        # TITLE
        # ====================================================

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


        # ====================================================
        # WORKSPACE
        # ====================================================

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


        # ====================================================
        # COMPONENT PANEL
        # ====================================================

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
        # MEDIAPIPE IMAGE
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


        # ====================================================
        # CURSOR
        # ====================================================

        cursor_x = None
        cursor_y = None

        current_menu_component = None

        is_pinching = False


        # ====================================================
        # HAND DETECTED
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ------------------------------------------------
            # INDEX FINGER
            # ------------------------------------------------

            index_tip = hand[8]


            # ------------------------------------------------
            # THUMB
            # ------------------------------------------------

            thumb_tip = hand[4]


            # ------------------------------------------------
            # PINCH DISTANCE
            # ------------------------------------------------

            dx = index_tip.x - thumb_tip.x
            dy = index_tip.y - thumb_tip.y

            pinch_distance = math.sqrt(
                dx * dx + dy * dy
            )


            is_pinching = (
                pinch_distance < PINCH_THRESHOLD
            )


            # ------------------------------------------------
            # TARGET CURSOR
            # ------------------------------------------------

            target_x = index_tip.x * WIDTH
            target_y = index_tip.y * HEIGHT


            # ------------------------------------------------
            # SMOOTH CURSOR
            # ------------------------------------------------

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


            # Keep cursor inside board

            cursor_x = max(
                0,
                min(WIDTH - 1, cursor_x)
            )

            cursor_y = max(
                0,
                min(HEIGHT - 1, cursor_y)
            )


            # =================================================
            # PINCH / DRAG
            # =================================================

            if is_pinching:

                # ---------------------------------------------
                # Start dragging
                # ---------------------------------------------

                if not dragging:

                    component = find_placed_component(
                        cursor_x,
                        cursor_y
                    )

                    if component is not None:

                        dragging = True

                        dragged_component = component

                        grab_offset_x = (
                            cursor_x -
                            component["x"]
                        )

                        grab_offset_y = (
                            cursor_y -
                            component["y"]
                        )

                        print(
                            "GRABBED:",
                            component["name"]
                        )


                # ---------------------------------------------
                # Move component
                # ---------------------------------------------

                if dragging and dragged_component is not None:

                    new_x = (
                        cursor_x -
                        grab_offset_x
                    )

                    new_y = (
                        cursor_y -
                        grab_offset_y
                    )

                    # Keep component inside workspace

                    new_x = max(
                        35,
                        min(665, new_x)
                    )

                    new_y = max(
                        155,
                        min(525, new_y)
                    )

                    dragged_component["x"] = new_x
                    dragged_component["y"] = new_y


            # =================================================
            # RELEASE
            # =================================================

            else:

                if dragging and dragged_component is not None:

                    print(
                        "RELEASED:",
                        dragged_component["name"]
                    )

                dragging = False
                dragged_component = None


            # =================================================
            # POINTING AT MENU
            # =================================================

            current_menu_component = find_component(
                cursor_x,
                cursor_y
            )


            # =================================================
            # PLACEMENT STABILITY
            # =================================================

            if not is_pinching:

                if (
                    current_menu_component ==
                    last_menu_component
                ):

                    stable_frames += 1

                else:

                    stable_frames = 0

                    last_menu_component = (
                        current_menu_component
                    )


                # ---------------------------------------------
                # PLACE COMPONENT
                # ---------------------------------------------

                if (
                    current_menu_component is not None
                    and
                    stable_frames >= REQUIRED_FRAMES
                    and
                    cooldown == 0
                ):

                    number = len(placed)

                    # Arrange components in workspace

                    x = 100 + (
                        number % 3
                    ) * 240

                    y = 190 + (
                        number // 3
                    ) * 140


                    # Only place if there is space

                    if y <= 500:

                        new_component = {
                            "name": current_menu_component,
                            "x": x,
                            "y": y
                        }

                        placed.append(
                            new_component
                        )

                        print(
                            "PLACED:",
                            current_menu_component
                        )


                    stable_frames = 0
                    last_menu_component = None

                    cooldown = 40


            # =================================================
            # DRAW HAND LANDMARKS ON CAMERA
            # =================================================

            for point in hand:

                px = int(
                    point.x *
                    camera_width
                )

                py = int(
                    point.y *
                    camera_height
                )

                cv2.circle(
                    camera,
                    (px, py),
                    3,
                    (0, 255, 0),
                    -1
                )


            # ------------------------------------------------
            # INDEX FINGER
            # ------------------------------------------------

            cv2.circle(
                camera,
                (
                    int(
                        index_tip.x *
                        camera_width
                    ),
                    int(
                        index_tip.y *
                        camera_height
                    )
                ),
                10,
                (0, 0, 255),
                3
            )


            # ------------------------------------------------
            # THUMB
            # ------------------------------------------------

            cv2.circle(
                camera,
                (
                    int(
                        thumb_tip.x *
                        camera_width
                    ),
                    int(
                        thumb_tip.y *
                        camera_height
                    )
                ),
                10,
                (255, 0, 0),
                3
            )


            # ------------------------------------------------
            # PINCH LINE
            # ------------------------------------------------

            cv2.line(
                camera,
                (
                    int(
                        index_tip.x *
                        camera_width
                    ),
                    int(
                        index_tip.y *
                        camera_height
                    )
                ),
                (
                    int(
                        thumb_tip.x *
                        camera_width
                    ),
                    int(
                        thumb_tip.y *
                        camera_height
                    )
                ),
                (255, 255, 0),
                2
            )


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            smooth_x = None
            smooth_y = None

            last_menu_component = None
            stable_frames = 0

            dragging = False
            dragged_component = None


        # ====================================================
        # COOLDOWN
        # ====================================================

        if cooldown > 0:

            cooldown -= 1


        # ====================================================
        # DRAW MENU
        # ====================================================

        for name, x1, y1, x2, y2 in menu:

            # Highlight menu item

            if name == current_menu_component:

                background = (255, 245, 190)
                border = (0, 190, 255)
                thickness = 5

            else:

                background = (255, 255, 255)
                border = (60, 60, 60)
                thickness = 2


            cv2.rectangle(
                board,
                (x1, y1),
                (x2, y2),
                background,
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
                (x1 + 55, y1 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (20, 20, 20),
                2
            )


        # ====================================================
        # DRAW PLACED COMPONENTS
        # ====================================================

        for component in placed:

            selected = (
                component == dragged_component
            )

            draw_component(
                board,
                component["name"],
                int(component["x"]),
                int(component["y"]),
                selected
            )


        # ====================================================
        # CURSOR
        # ====================================================

        if cursor_x is not None:

            if is_pinching:

                cursor_color = (255, 0, 255)

            else:

                cursor_color = (0, 0, 255)


            cv2.circle(
                board,
                (cursor_x, cursor_y),
                14,
                cursor_color,
                3
            )

            cv2.circle(
                board,
                (cursor_x, cursor_y),
                4,
                cursor_color,
                -1
            )


        # ====================================================
        # STATUS BAR
        # ====================================================

        if dragging:

            message = (
                "GRABBING "
                + dragged_component["name"]
                + " - move your hand"
            )

        elif is_pinching:

            message = (
                "PINCH DETECTED"
            )

        elif current_menu_component is not None:

            seconds = stable_frames / 30

            message = (
                "Holding "
                + current_menu_component
                + " : "
                + str(round(seconds, 1))
                + " / 1.0 sec"
            )

        else:

            message = (
                "Point at a component"
            )


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
            "POINT + HOLD = PLACE",
            (50, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 80, 80),
            1
        )

        cv2.putText(
            board,
            "PINCH = GRAB / MOVE",
            (50, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 80, 80),
            1
        )

        cv2.putText(
            board,
            "RELEASE PINCH = DROP",
            (50, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 80, 80),
            1
        )


        # ====================================================
        # SHOW WINDOWS
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