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

# Your model is inside Gesture/models
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
# DATA
# ============================================================

placed = []

# Each wire contains two component objects
wires = []

# First component selected for wiring
wire_start = None


# ============================================================
# FIND COMPONENT IN MENU
# ============================================================

def find_component(x, y):

    for name, x1, y1, x2, y2 in menu:

        if x1 <= x <= x2 and y1 <= y <= y2:
            return name

    return None


# ============================================================
# FIND PLACED COMPONENT
# ============================================================

def find_placed_component(x, y):

    for component in reversed(placed):

        cx = int(component["x"])
        cy = int(component["y"])

        if (
            cx <= x <= cx + 180
            and
            cy <= y <= cy + 90
        ):
            return component

    return None


# ============================================================
# GET COMPONENT TERMINALS
# ============================================================

def get_terminals(component):

    x = int(component["x"])
    y = int(component["y"])

    left = (
        x,
        y + 45
    )

    right = (
        x + 180,
        y + 45
    )

    return left, right


# ============================================================
# DRAW WIRE
# ============================================================

def draw_wire(board, component1, component2):

    left1, right1 = get_terminals(component1)
    left2, right2 = get_terminals(component2)

    possibilities = [
        (left1, left2),
        (left1, right2),
        (right1, left2),
        (right1, right2)
    ]

    def distance(pair):

        x1, y1 = pair[0]
        x2, y2 = pair[1]

        return (
            (x1 - x2) ** 2
            +
            (y1 - y2) ** 2
        )

    start, end = min(
        possibilities,
        key=distance
    )

    # Wire
    cv2.line(
        board,
        start,
        end,
        (40, 40, 40),
        5
    )

    # Connection points
    cv2.circle(
        board,
        start,
        8,
        (0, 140, 255),
        -1
    )

    cv2.circle(
        board,
        end,
        8,
        (0, 140, 255),
        -1
    )


# ============================================================
# DRAW COMPONENT
# ============================================================

def draw_component(
    board,
    name,
    x,
    y,
    selected=False,
    wire_selected=False
):

    x = int(x)
    y = int(y)

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    if wire_selected:

        background = (220, 255, 220)
        border = (0, 180, 0)
        thickness = 5

    elif selected:

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
    # Battery
    # --------------------------------------------------------

    if name == "BATTERY":

        color = (70, 70, 220)

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
            (x + 25, y + 45),
            (x + 50, y + 45),
            color,
            3
        )

        cv2.line(
            board,
            (x + 75, y + 45),
            (x + 105, y + 45),
            color,
            3
        )

    # --------------------------------------------------------
    # Resistor
    # --------------------------------------------------------

    elif name == "RESISTOR":

        color = (40, 130, 220)

        points = np.array(
            [
                [x + 25, y + 45],
                [x + 45, y + 25],
                [x + 65, y + 65],
                [x + 85, y + 25],
                [x + 105, y + 65],
                [x + 125, y + 45]
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

    # --------------------------------------------------------
    # LED
    # --------------------------------------------------------

    elif name == "LED":

        color = (40, 180, 90)

        cv2.circle(
            board,
            (x + 80, y + 45),
            25,
            color,
            3
        )

        cv2.line(
            board,
            (x + 25, y + 45),
            (x + 55, y + 45),
            color,
            3
        )

        cv2.line(
            board,
            (x + 105, y + 45),
            (x + 135, y + 45),
            color,
            3
        )

        # LED arrows
        cv2.arrowedLine(
            board,
            (x + 70, y + 15),
            (x + 55, y + 5),
            color,
            2
        )

        cv2.arrowedLine(
            board,
            (x + 90, y + 15),
            (x + 75, y + 5),
            color,
            2
        )

    # --------------------------------------------------------
    # Capacitor
    # --------------------------------------------------------

    elif name == "CAPACITOR":

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
            (x + 95, y + 20),
            (x + 95, y + 70),
            color,
            5
        )

        cv2.line(
            board,
            (x + 25, y + 45),
            (x + 65, y + 45),
            color,
            3
        )

        cv2.line(
            board,
            (x + 95, y + 45),
            (x + 135, y + 45),
            color,
            3
        )

    # --------------------------------------------------------
    # Name
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
# SMOOTHING
# ============================================================

smooth_x = None
smooth_y = None

SMOOTHING = 0.20


# ============================================================
# PLACEMENT
# ============================================================

last_menu_component = None
stable_frames = 0

REQUIRED_FRAMES = 30
cooldown = 0


# ============================================================
# PINCH / DRAG
# ============================================================

dragging = False
dragged_component = None

grab_offset_x = 0
grab_offset_y = 0

PINCH_THRESHOLD = 0.055

# Used to detect whether pinch was a click or drag
pinch_start_x = None
pinch_start_y = None

PINCH_MOVE_LIMIT = 35


# ============================================================
# FRAME
# ============================================================

frame_number = 0


# ============================================================
# MAIN LOOP
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
        # WHITE BOARD
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


        # ====================================================
        # CURSOR VARIABLES
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

            index_tip = hand[8]
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
            # TARGET POSITION
            # ------------------------------------------------

            target_x = index_tip.x * WIDTH
            target_y = index_tip.y * HEIGHT


            # ------------------------------------------------
            # SMOOTH POSITION
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

            cursor_x = max(
                0,
                min(WIDTH - 1, cursor_x)
            )

            cursor_y = max(
                0,
                min(HEIGHT - 1, cursor_y)
            )


            # =================================================
            # PINCH START
            # =================================================

            if is_pinching:

                if pinch_start_x is None:

                    pinch_start_x = cursor_x
                    pinch_start_y = cursor_y

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
                            "PINCH START:",
                            component["name"]
                        )


                # =================================================
                # MOVE COMPONENT
                # =================================================

                if dragging and dragged_component is not None:

                    new_x = (
                        cursor_x -
                        grab_offset_x
                    )

                    new_y = (
                        cursor_y -
                        grab_offset_y
                    )

                    # Workspace boundaries

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
            # PINCH RELEASE
            # =================================================

            else:

                if pinch_start_x is not None:

                    if cursor_x is not None:

                        move_distance = math.sqrt(
                            (
                                cursor_x -
                                pinch_start_x
                            ) ** 2
                            +
                            (
                                cursor_y -
                                pinch_start_y
                            ) ** 2
                        )

                    else:

                        move_distance = 9999


                    # -----------------------------------------
                    # SHORT PINCH
                    # -----------------------------------------

                    if move_distance < PINCH_MOVE_LIMIT:

                        component = find_placed_component(
                            pinch_start_x,
                            pinch_start_y
                        )

                        if component is not None:

                            # ---------------------------------
                            # FIRST WIRE COMPONENT
                            # ---------------------------------

                            if wire_start is None:

                                wire_start = component

                                print(
                                    "WIRE START:",
                                    component["name"]
                                )


                            # ---------------------------------
                            # SECOND WIRE COMPONENT
                            # ---------------------------------

                            elif component != wire_start:

                                already_exists = False

                                for wire in wires:

                                    if (
                                        (
                                            wire[0] ==
                                            wire_start
                                            and
                                            wire[1] ==
                                            component
                                        )
                                        or
                                        (
                                            wire[0] ==
                                            component
                                            and
                                            wire[1] ==
                                            wire_start
                                        )
                                    ):

                                        already_exists = True
                                        break


                                if not already_exists:

                                    wires.append(
                                        (
                                            wire_start,
                                            component
                                        )
                                    )

                                    print(
                                        "WIRE CREATED:",
                                        wire_start["name"],
                                        "->",
                                        component["name"]
                                    )

                                wire_start = None


                    # -----------------------------------------
                    # FINISH DRAG
                    # -----------------------------------------

                    elif dragging:

                        if dragged_component is not None:

                            print(
                                "MOVED:",
                                dragged_component["name"]
                            )


                dragging = False
                dragged_component = None

                pinch_start_x = None
                pinch_start_y = None


            # =================================================
            # MENU COMPONENT
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
                # PLACE
                # ---------------------------------------------

                if (
                    current_menu_component is not None
                    and
                    stable_frames >= REQUIRED_FRAMES
                    and
                    cooldown == 0
                ):

                    number = len(placed)

                    x = (
                        100 +
                        (number % 3) * 240
                    )

                    y = (
                        190 +
                        (number // 3) * 140
                    )


                    if y <= 500:

                        new_component = {
                            "name":
                                current_menu_component,

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
            # DRAW HAND LANDMARKS
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


            # Index finger
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


            # Thumb
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


            # Pinch line
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

            pinch_start_x = None
            pinch_start_y = None


        # ====================================================
        # COOLDOWN
        # ====================================================

        if cooldown > 0:

            cooldown -= 1


        # ====================================================
        # DRAW MENU
        # ====================================================

        for name, x1, y1, x2, y2 in menu:

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
        # DRAW WIRES
        # ====================================================

        for wire in wires:

            draw_wire(
                board,
                wire[0],
                wire[1]
            )


        # ====================================================
        # DRAW COMPONENTS
        # ====================================================

        for component in placed:

            selected = (
                component == dragged_component
            )

            wire_selected = (
                component == wire_start
            )

            draw_component(
                board,
                component["name"],
                component["x"],
                component["y"],
                selected,
                wire_selected
            )


        # ====================================================
        # CURSOR
        # ====================================================

        if cursor_x is not None:

            if is_pinching:

                cursor_color = (255, 0, 255)

            elif wire_start is not None:

                cursor_color = (0, 180, 0)

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
        # STATUS MESSAGE
        # ====================================================

        if dragging and dragged_component is not None:

            message = (
                "MOVING "
                + dragged_component["name"]
                + " - release pinch to drop"
            )

        elif wire_start is not None:

            message = (
                "WIRE START: "
                + wire_start["name"]
                + " | Pinch another component"
            )

        elif is_pinching:

            message = "PINCH DETECTED"

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
            "POINT + HOLD = PLACE",
            (50, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 80, 80),
            1
        )

        cv2.putText(
            board,
            "PINCH + MOVE = MOVE",
            (50, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 80, 80),
            1
        )

        cv2.putText(
            board,
            "SHORT PINCH = CONNECT",
            (50, 245),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
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