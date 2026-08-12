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

MODEL_PATH = "Gesture/models/hand_landmarker.task"


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


# ============================================================
# DATA
# ============================================================

placed = []

wires = []

wire_start = None


# ============================================================
# FIND MENU COMPONENT
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
# GET TERMINAL
# ============================================================

def get_terminal(component, side):

    x = int(component["x"])
    y = int(component["y"])

    if side == "left":

        return (
            x,
            y + 45
        )

    else:

        return (
            x + 180,
            y + 45
        )


# ============================================================
# FIND BEST TERMINALS
# ============================================================

def get_wire_points(component1, component2):

    x1 = component1["x"]
    x2 = component2["x"]

    # If second component is to the right
    if x2 >= x1:

        start = get_terminal(
            component1,
            "right"
        )

        end = get_terminal(
            component2,
            "left"
        )

    else:

        start = get_terminal(
            component1,
            "left"
        )

        end = get_terminal(
            component2,
            "right"
        )

    return start, end


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
    # Colors
    # --------------------------------------------------------

    if selected:

        background = (210, 245, 255)
        border = (0, 150, 255)
        thickness = 4

    elif wire_selected:

        background = (230, 255, 230)
        border = (0, 180, 80)
        thickness = 4

    else:

        background = (245, 245, 245)
        border = (50, 50, 50)
        thickness = 2


    # --------------------------------------------------------
    # Box
    # --------------------------------------------------------

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
    # Component symbol
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
# DRAW WIRE
# ============================================================

def draw_wire(board, component1, component2):

    start, end = get_wire_points(
        component1,
        component2
    )

    # --------------------------------------------------------
    # Connection line
    # --------------------------------------------------------

    cv2.line(
        board,
        start,
        end,
        (40, 40, 40),
        5
    )


    # --------------------------------------------------------
    # Connection dots
    # --------------------------------------------------------

    cv2.circle(
        board,
        start,
        8,
        (0, 120, 255),
        -1
    )

    cv2.circle(
        board,
        end,
        8,
        (0, 120, 255),
        -1
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


# ============================================================
# PINCH
# ============================================================

dragging = False

dragged_component = None

grab_offset_x = 0
grab_offset_y = 0

PINCH_THRESHOLD = 0.055


# ============================================================
# COOLDOWN
# ============================================================

cooldown = 0


# ============================================================
# FRAME
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


        camera = cv2.flip(
            camera,
            1
        )

        camera_height, camera_width, _ = camera.shape


        # ====================================================
        # BOARD
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
        # MENU
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


        cursor_x = None
        cursor_y = None

        current_menu_component = None

        is_pinching = False


        # ====================================================
        # HAND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            index_tip = hand[8]

            thumb_tip = hand[4]


            # ------------------------------------------------
            # PINCH
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
            # CURSOR
            # ------------------------------------------------

            target_x = index_tip.x * WIDTH
            target_y = index_tip.y * HEIGHT


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
            # GRAB / MOVE
            # =================================================

            if is_pinching:

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


                if (
                    dragging
                    and
                    dragged_component is not None
                ):

                    new_x = (
                        cursor_x -
                        grab_offset_x
                    )

                    new_y = (
                        cursor_y -
                        grab_offset_y
                    )

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


            else:

                if (
                    dragging
                    and
                    dragged_component is not None
                ):

                    print(
                        "RELEASED:",
                        dragged_component["name"]
                    )

                dragging = False
                dragged_component = None


            # =================================================
            # MENU POINTING
            # =================================================

            current_menu_component = find_component(
                cursor_x,
                cursor_y
            )


            # =================================================
            # PLACE COMPONENT
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


                if (
                    current_menu_component is not None
                    and
                    stable_frames >= REQUIRED_FRAMES
                    and
                    cooldown == 0
                ):

                    number = len(placed)

                    x = 100 + (
                        number % 3
                    ) * 240

                    y = 190 + (
                        number // 3
                    ) * 140


                    if y <= 500:

                        component = {
                            "name":
                                current_menu_component,

                            "x": x,

                            "y": y
                        }

                        placed.append(
                            component
                        )

                        print(
                            "PLACED:",
                            current_menu_component
                        )


                    stable_frames = 0

                    last_menu_component = None

                    cooldown = 40


            # =================================================
            # CAMERA LANDMARKS
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
                component ==
                dragged_component
            )

            wire_selected = (
                component ==
                wire_start
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
        # CONNECTION DOT FOR WIRE SELECTION
        # ====================================================

        if wire_start is not None:

            left, right = get_wire_points(
                wire_start,
                wire_start
            )

            terminal = get_terminal(
                wire_start,
                "right"
            )

            cv2.circle(
                board,
                terminal,
                11,
                (0, 180, 255),
                3
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
        # STATUS
        # ====================================================

        if dragging:

            message = (
                "GRABBING "
                + dragged_component["name"]
                + " - move your hand"
            )

        elif wire_start is not None:

            message = (
                "WIRE START: "
                + wire_start["name"]
                + " - point at another component and press W"
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
                "Point at component | W = connect components"
            )


        # ====================================================
        # STATUS BAR
        # ====================================================

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
            0.50,
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
            "W = CONNECT TWO COMPONENTS",
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
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        # ----------------------------------------------------
        # W = START / COMPLETE WIRE
        # ----------------------------------------------------

        if key == ord("w"):

            if cursor_x is not None:

                component = find_placed_component(
                    cursor_x,
                    cursor_y
                )

                if component is not None:

                    # First component

                    if wire_start is None:

                        wire_start = component

                        print(
                            "WIRE START:",
                            component["name"]
                        )


                    # Second component

                    else:

                        if component != wire_start:

                            # Check duplicate

                            duplicate = False

                            for existing in wires:

                                if (
                                    (
                                        existing[0] ==
                                        wire_start
                                        and
                                        existing[1] ==
                                        component
                                    )
                                    or
                                    (
                                        existing[0] ==
                                        component
                                        and
                                        existing[1] ==
                                        wire_start
                                    )
                                ):

                                    duplicate = True

                                    break


                            if not duplicate:

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

                            else:

                                print(
                                    "WIRE ALREADY EXISTS"
                                )


                            wire_start = None


        # ----------------------------------------------------
        # ESC = CANCEL WIRE
        # ----------------------------------------------------

        if key == 27:

            wire_start = None

            print(
                "WIRE CANCELLED"
            )


        # ----------------------------------------------------
        # Q = QUIT
        # ----------------------------------------------------

        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print(
    "Circuit Vision closed."
)