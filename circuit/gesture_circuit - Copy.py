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

    # Faster response
    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.60
)


# ============================================================
# COMPONENT MENU
# SMALL COMPONENT PANEL
# ============================================================

menu = [
    ("BATTERY", 1000, 165, 1165, 235),
    ("RESISTOR", 1000, 260, 1165, 330),
    ("LED", 1000, 355, 1165, 425),
    ("CAPACITOR", 1000, 450, 1165, 520)
]


# ============================================================
# CIRCUIT DATA
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
# TERMINALS
# ============================================================

def get_terminals(component):

    x = int(component["x"])
    y = int(component["y"])

    left = (x, y + 45)
    right = (x + 180, y + 45)

    return left, right


# ============================================================
# CIRCUIT VALIDATION
# ============================================================

def check_circuit():

    names = []

    for component in placed:

        if component["name"] not in names:
            names.append(component["name"])

    if "BATTERY" not in names:
        return False, "Add Battery"

    if "RESISTOR" not in names:
        return False, "Add Resistor"

    if "LED" not in names:
        return False, "Add LED"

    battery_resistor = False
    resistor_led = False

    for wire in wires:

        name1 = wire[0]["name"]
        name2 = wire[1]["name"]

        pair = {name1, name2}

        if pair == {"BATTERY", "RESISTOR"}:
            battery_resistor = True

        if pair == {"RESISTOR", "LED"}:
            resistor_led = True

    if not battery_resistor:
        return False, "Connect Battery to Resistor"

    if not resistor_led:
        return False, "Connect Resistor to LED"

    return True, "CIRCUIT COMPLETE"


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

    def wire_distance(pair):

        x1, y1 = pair[0]
        x2, y2 = pair[1]

        return (
            (x1 - x2) ** 2 +
            (y1 - y2) ** 2
        )

    start, end = min(
        possibilities,
        key=wire_distance
    )

    cv2.line(
        board,
        start,
        end,
        (40, 40, 40),
        5
    )

    cv2.circle(
        board,
        start,
        7,
        (0, 140, 255),
        -1
    )

    cv2.circle(
        board,
        end,
        7,
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
    wire_selected=False,
    led_on=False
):

    x = int(x)
    y = int(y)

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


    # ========================================================
    # BATTERY
    # ========================================================

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


    # ========================================================
    # RESISTOR
    # ========================================================

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


    # ========================================================
    # LED
    # ========================================================

    elif name == "LED":

        if led_on:

            for radius in [35, 30, 25]:

                cv2.circle(
                    board,
                    (x + 80, y + 45),
                    radius,
                    (100, 255, 100),
                    3
                )

            color = (0, 255, 80)

            cv2.circle(
                board,
                (x + 80, y + 45),
                23,
                color,
                -1
            )

            cv2.circle(
                board,
                (x + 80, y + 45),
                25,
                (0, 100, 0),
                3
            )

        else:

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


    # ========================================================
    # CAPACITOR
    # ========================================================

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


    # ========================================================
    # NAME
    # ========================================================

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
# SAME ORIGINAL CAMERA WINDOW
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# ============================================================
# CURSOR
# FASTER + SMOOTHER
# ============================================================

smooth_x = None
smooth_y = None

# Increased from 0.20
SMOOTHING = 0.40


# ============================================================
# PLACEMENT
# ============================================================

last_menu_component = None
stable_frames = 0

# Faster placement
REQUIRED_FRAMES = 20

cooldown = 0


# ============================================================
# DRAGGING
# ============================================================

dragging = False
dragged_component = None

grab_offset_x = 0
grab_offset_y = 0


# ============================================================
# PINCH
# ============================================================

PINCH_ON = 0.050
PINCH_OFF = 0.065

previous_pinch = False

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
            (985, 620),
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
        # SMALL COMPONENT PANEL
        # ====================================================

        cv2.rectangle(
            board,
            (990, 110),
            (1180, 620),
            (240, 240, 240),
            -1
        )

        cv2.rectangle(
            board,
            (990, 110),
            (1180, 620),
            (150, 150, 150),
            2
        )

        cv2.putText(
            board,
            "COMPONENTS",
            (1000, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (20, 20, 20),
            1
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


            # =================================================
            # PINCH DETECTION
            # =================================================

            dx = (
                index_tip.x -
                thumb_tip.x
            )

            dy = (
                index_tip.y -
                thumb_tip.y
            )

            pinch_distance = math.sqrt(
                dx * dx +
                dy * dy
            )


            # Hysteresis prevents flickering
            if previous_pinch:

                is_pinching = (
                    pinch_distance <
                    PINCH_OFF
                )

            else:

                is_pinching = (
                    pinch_distance <
                    PINCH_ON
                )


            previous_pinch = is_pinching


            # =================================================
            # FAST CURSOR
            # =================================================

            target_x = (
                index_tip.x *
                WIDTH
            )

            target_y = (
                index_tip.y *
                HEIGHT
            )


            if smooth_x is None:

                smooth_x = target_x
                smooth_y = target_y

            else:

                smooth_x += (
                    target_x -
                    smooth_x
                ) * SMOOTHING

                smooth_y += (
                    target_y -
                    smooth_y
                ) * SMOOTHING


            cursor_x = int(smooth_x)
            cursor_y = int(smooth_y)


            cursor_x = max(
                0,
                min(
                    WIDTH - 1,
                    cursor_x
                )
            )

            cursor_y = max(
                0,
                min(
                    HEIGHT - 1,
                    cursor_y
                )
            )


            # =================================================
            # PINCH START
            # =================================================

            if is_pinching:

                if pinch_start_x is None:

                    pinch_start_x = cursor_x
                    pinch_start_y = cursor_y


                    component = (
                        find_placed_component(
                            cursor_x,
                            cursor_y
                        )
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


                # =================================================
                # MOVE
                # =================================================

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


                    # Workspace limits

                    new_x = max(
                        35,
                        min(
                            800,
                            new_x
                        )
                    )

                    new_y = max(
                        155,
                        min(
                            525,
                            new_y
                        )
                    )


                    dragged_component["x"] = new_x
                    dragged_component["y"] = new_y


            # =================================================
            # RELEASE
            # =================================================

            else:

                if pinch_start_x is not None:

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


                    # Short pinch = connect

                    if (
                        move_distance <
                        PINCH_MOVE_LIMIT
                    ):

                        component = (
                            find_placed_component(
                                pinch_start_x,
                                pinch_start_y
                            )
                        )


                        if component is not None:

                            if wire_start is None:

                                wire_start = component

                                print(
                                    "WIRE START:",
                                    component["name"]
                                )


                            elif component != wire_start:

                                exists = False


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

                                        exists = True
                                        break


                                if not exists:

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


                dragging = False
                dragged_component = None

                pinch_start_x = None
                pinch_start_y = None


            # =================================================
            # MENU
            # =================================================

            current_menu_component = (
                find_component(
                    cursor_x,
                    cursor_y
                )
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
                    stable_frames >=
                    REQUIRED_FRAMES
                    and
                    cooldown == 0
                ):

                    number = len(placed)


                    x = (
                        100 +
                        (number % 4) *
                        200
                    )

                    y = (
                        180 +
                        (number // 4) *
                        140
                    )


                    if y <= 500:

                        placed.append(
                            {
                                "name":
                                    current_menu_component,

                                "x":
                                    x,

                                "y":
                                    y
                            }
                        )


                        print(
                            "PLACED:",
                            current_menu_component
                        )


                    stable_frames = 0

                    last_menu_component = None

                    cooldown = 30


            # =================================================
            # CAMERA HAND LANDMARKS
            # SAME AS ORIGINAL
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


            # INDEX

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


            # THUMB

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


        # ====================================================
        # NO HAND
        # ====================================================

        else:

            smooth_x = None
            smooth_y = None

            previous_pinch = False

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
        # CHECK CIRCUIT
        # ====================================================

        circuit_complete, circuit_message = (
            check_circuit()
        )


        # ====================================================
        # DRAW MENU
        # ====================================================

        for name, x1, y1, x2, y2 in menu:

            if name == current_menu_component:

                background = (
                    255,
                    245,
                    190
                )

                border = (
                    0,
                    190,
                    255
                )

                thickness = 4

            else:

                background = (
                    255,
                    255,
                    255
                )

                border = (
                    60,
                    60,
                    60
                )

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
                (x1 + 25, y1 + 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
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

            led_on = (
                circuit_complete
                and
                component["name"] ==
                "LED"
            )

            draw_component(
                board,
                component["name"],
                component["x"],
                component["y"],
                selected,
                wire_selected,
                led_on
            )


        # ====================================================
        # CURSOR
        # ====================================================

        if cursor_x is not None:

            if is_pinching:

                cursor_color = (
                    255,
                    0,
                    255
                )

            elif wire_start is not None:

                cursor_color = (
                    0,
                    180,
                    0
                )

            else:

                cursor_color = (
                    0,
                    0,
                    255
                )


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

        if circuit_complete:

            status_text = (
                "CIRCUIT COMPLETE - LED ON"
            )

            status_color = (
                0,
                180,
                0
            )

        else:

            status_text = (
                "CIRCUIT INCOMPLETE"
            )

            status_color = (
                0,
                0,
                220
            )


        cv2.putText(
            board,
            status_text,
            (600, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            status_color,
            2
        )


        # ====================================================
        # STATUS MESSAGE
        # ====================================================

        if (
            dragging
            and
            dragged_component is not None
        ):

            message = (
                "MOVING "
                +
                dragged_component["name"]
                +
                " - release pinch"
            )

        elif wire_start is not None:

            message = (
                "WIRE START: "
                +
                wire_start["name"]
                +
                " | pinch another component"
            )

        elif current_menu_component is not None:

            seconds = (
                stable_frames /
                30
            )

            message = (
                "Holding "
                +
                current_menu_component
                +
                " : "
                +
                str(
                    round(
                        seconds,
                        1
                    )
                )
                +
                " / 0.7 sec"
            )

        else:

            message = circuit_message


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
            0.55,
            (255, 255, 255),
            2
        )


        # ====================================================
        # WINDOWS
        # IMPORTANT:
        # CAMERA IS SEPARATE LIKE YOUR ORIGINAL CODE
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

