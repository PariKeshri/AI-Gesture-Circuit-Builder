import cv2
import mediapipe as mp
import numpy as np
import math
import copy

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
    ("BATTERY", 1000, 150, 1170, 188),
    ("330 OHM", 1000, 192, 1170, 230),
    ("1 kOHM", 1000, 234, 1170, 272),
    ("2.2 kOHM", 1000, 276, 1170, 314),
    ("3.3 kOHM", 1000, 318, 1170, 356),
    ("10 kOHM", 1000, 360, 1170, 398),
    ("100 kOHM", 1000, 402, 1170, 440),
    ("LED", 1000, 444, 1170, 482),
    ("CAPACITOR", 1000, 486, 1170, 524),
    ("DIODE", 1000, 528, 1170, 566),
    ("CURRENT SOURCE", 1000, 570, 1170, 608)
]


# ============================================================
# CIRCUIT DATA
# ============================================================

placed = []
wires = []

wire_start = None
wire_start_terminal = None

# ============================================================
# UNDO HISTORY
# ============================================================

history = []


def save_state():
    """Save a complete circuit state for one-step undo."""
    placed_copy = [
        {
            "name": component["name"],
            "x": component["x"],
            "y": component["y"]
        }
        for component in placed
    ]

    wires_copy = []

    for wire in wires:
        try:
            index1 = placed.index(wire[0])
            index2 = placed.index(wire[1])
        except ValueError:
            continue

        wires_copy.append(
            (
                index1,
                index2,
                wire[2],
                wire[3]
            )
        )

    history.append(
        (
            placed_copy,
            wires_copy
        )
    )

    # Keep memory bounded.
    if len(history) > 50:
        history.pop(0)


def undo_action():
    global wire_start
    global wire_start_terminal
    global dragging
    global dragged_component
    global pinch_start_x
    global pinch_start_y

    if not history:
        print("UNDO: NOTHING TO UNDO")
        return False

    placed_copy, wires_copy = history.pop()

    placed.clear()

    for component in placed_copy:
        placed.append(
            {
                "name": component["name"],
                "x": component["x"],
                "y": component["y"]
            }
        )

    wires.clear()

    for index1, index2, terminal1, terminal2 in wires_copy:
        if (
            0 <= index1 < len(placed)
            and
            0 <= index2 < len(placed)
            and
            terminal1 in (0, 1)
            and
            terminal2 in (0, 1)
        ):
            wires.append(
                (
                    placed[index1],
                    placed[index2],
                    terminal1,
                    terminal2
                )
            )

    wire_start = None
    wire_start_terminal = None
    dragging = False
    dragged_component = None
    pinch_start_x = None
    pinch_start_y = None

    print("UNDO: ACTION REVERTED")
    return True


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

# ============================================================
# TERMINALS
# ============================================================

def get_terminals(component):

    x = int(component["x"])
    y = int(component["y"])

    # All supported components are two-terminal components.
    return (
        (x, y + 45),
        (x + 180, y + 45)
    )


def find_terminal(x, y):

    TERMINAL_RADIUS = 25

    for component in reversed(placed):

        terminals = get_terminals(component)

        for index, terminal in enumerate(terminals):

            tx, ty = terminal

            distance = math.sqrt(
                (x - tx) ** 2 +
                (y - ty) ** 2
            )

            if distance <= TERMINAL_RADIUS:

                return component, index

    return None, None

# ============================================================
# CIRCUIT VALIDATION
# ============================================================

def check_circuit():

    resistor_names = {
        "330 OHM",
        "1 kOHM",
        "2.2 kOHM",
        "3.3 kOHM",
        "10 kOHM",
        "100 kOHM"
    }

    source_names = {
        "BATTERY",
        "CURRENT SOURCE"
    }

    load_names = {
        "LED",
        "CAPACITOR",
        "DIODE"
    }

    names = {
        component["name"]
        for component in placed
    }

    if not placed:
        return False, "Place Components"

    if "BATTERY" not in names and "CURRENT SOURCE" not in names:
        return False, "Add a Source"

    if not names.intersection(resistor_names):
        return False, "Add Resistor"

    # --------------------------------------------------------
    # COMPONENT-LEVEL CONNECTION GRAPH
    # --------------------------------------------------------

    graph = {
        id(component): set()
        for component in placed
    }

    for wire in wires:

        component1 = wire[0]
        component2 = wire[1]

        if (
            component1 not in placed
            or
            component2 not in placed
        ):
            continue

        graph[id(component1)].add(id(component2))
        graph[id(component2)].add(id(component1))

    def neighbours(component):
        result = []

        for component_id in graph.get(id(component), set()):

            for candidate in placed:

                if id(candidate) == component_id:
                    result.append(candidate)
                    break

        return result

    def connected(a, b):
        return id(b) in graph.get(id(a), set())

    def all_connected():

        start_component = placed[0]
        visited = set()

        stack = [id(start_component)]

        while stack:

            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)

            for neighbour in graph.get(current, set()):
                if neighbour not in visited:
                    stack.append(neighbour)

        return len(visited) == len(placed)

    def is_closed_network():

        if not all_connected():
            return False

        # A closed component-level circuit has no
        # component with only one connection.
        for component in placed:

            if len(graph[id(component)]) < 2:
                return False

        return True

    def has_component(name):
        return any(
            component["name"] == name
            for component in placed
        )

    def components_with_name(name):
        return [
            component
            for component in placed
            if component["name"] == name
        ]

    def has_series_path(names_to_find):

        for start_component in placed:

            if start_component["name"] not in names_to_find:
                continue

            stack = [
                (start_component, {id(start_component)})
            ]

            while stack:

                current, visited = stack.pop()

                if current["name"] in names_to_find:
                    if names_to_find <= {
                        component["name"]
                        for component in placed
                    }:
                        return True

                for neighbour in neighbours(current):

                    if id(neighbour) not in visited:

                        stack.append(
                            (
                                neighbour,
                                visited | {id(neighbour)}
                            )
                        )

        return False

    # --------------------------------------------------------
    # 1. NORTON
    #
    # Current source and resistor share both nodes.
    #
    # Current Source 0 -> Resistor 0
    # Current Source 1 -> Resistor 1
    # OR the reversed terminal orientation.
    # --------------------------------------------------------

    current_sources = components_with_name(
        "CURRENT SOURCE"
    )

    resistors = [
        component
        for component in placed
        if component["name"] in resistor_names
    ]

    def direct_wire(c1, t1, c2, t2):

        for wire in wires:

            if (
                wire[0] is c1
                and
                wire[1] is c2
                and
                wire[2] == t1
                and
                wire[3] == t2
            ):
                return True

            if (
                wire[0] is c2
                and
                wire[1] is c1
                and
                wire[2] == t2
                and
                wire[3] == t1
            ):
                return True

        return False

    for source in current_sources:

        for resistor in resistors:

            parallel_same = (
                direct_wire(
                    source, 0,
                    resistor, 0
                )
                and
                direct_wire(
                    source, 1,
                    resistor, 1
                )
            )

            parallel_reversed = (
                direct_wire(
                    source, 0,
                    resistor, 1
                )
                and
                direct_wire(
                    source, 1,
                    resistor, 0
                )
            )

            if parallel_same or parallel_reversed:

                return True, "NORTON CIRCUIT"

    # --------------------------------------------------------
    # 2. PN JUNCTION
    #
    # Battery -> diode -> resistor
    # in a connected closed circuit.
    # --------------------------------------------------------

    diodes = components_with_name("DIODE")

    if diodes:

        for diode in diodes:

            diode_neighbours = neighbours(diode)

            has_source_neighbour = any(
                neighbour["name"] in source_names
                for neighbour in diode_neighbours
            )

            has_resistor_neighbour = any(
                neighbour["name"] in resistor_names
                for neighbour in diode_neighbours
            )

            if (
                has_source_neighbour
                and
                has_resistor_neighbour
                and
                is_closed_network()
            ):
                return True, "PN JUNCTION CIRCUIT"

    # --------------------------------------------------------
    # 3. SUPERPOSITION
    #
    # Two or more independent sources in one connected
    # network. Battery and Current Source are both sources.
    # --------------------------------------------------------

    sources = [
        component
        for component in placed
        if component["name"] in source_names
    ]

    if len(sources) >= 2:

        if (
            all_connected()
            and
            any(
                component["name"] in resistor_names
                for component in placed
            )
        ):
            return True, "SUPERPOSITION CIRCUIT"

    # --------------------------------------------------------
    # 4. THEVENIN
    #
    # One voltage source + resistor + load in a closed
    # connected network.
    # --------------------------------------------------------

    batteries = components_with_name("BATTERY")

    if (
        len(batteries) >= 1
        and
        len(current_sources) == 0
        and
        len(resistors) >= 2
        and
        "LED" in names
        and
        is_closed_network()
    ):
        return True, "THEVENIN CIRCUIT"

    # --------------------------------------------------------
    # 5. BASIC SERIES CIRCUIT
    #
    # One battery + one resistor + LED in a closed loop.
    # --------------------------------------------------------

    if (
        len(batteries) == 1
        and
        len(current_sources) == 0
        and
        len(resistors) == 1
        and
        "LED" in names
        and
        len(placed) == 3
        and
        is_closed_network()
    ):
        return True, "BASIC CIRCUIT"

    # --------------------------------------------------------
    # 6. BASIC CONNECTED CIRCUIT
    # --------------------------------------------------------

    if all_connected():

        return True, "CONNECTED CIRCUIT"

    return False, "Connect All Components"

def draw_wire(
    board,
    component1,
    terminal1,
    component2,
    terminal2
):

    if terminal1 not in (0, 1) or terminal2 not in (0, 1):
        return

    terminals1 = get_terminals(component1)
    terminals2 = get_terminals(component2)

    start = terminals1[terminal1]
    end = terminals2[terminal2]

    cv2.line(
        board,
        start,
        end,
        (40, 40, 40),
        5
    )

    cv2.circle(board, start, 7, (0, 140, 255), -1)
    cv2.circle(board, end, 7, (0, 140, 255), -1)


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

    # Soft card shadow + clean component card. Geometry remains unchanged.
    cv2.rectangle(
        board,
        (x + 4, y + 5),
        (x + 184, y + 95),
        (228, 231, 236),
        -1
    )

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

    # Terminal labels make the two-wire system easier to understand.
    terminals = get_terminals({"name": name, "x": x, "y": y})
    for idx, (tx, ty) in enumerate(terminals):
        cv2.circle(board, (int(tx), int(ty)), 9, (255, 255, 255), -1)
        cv2.circle(board, (int(tx), int(ty)), 7, (0, 150, 220), 2)
        cv2.putText(board, "T" + str(idx + 1), (int(tx) - 10, int(ty) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (75, 85, 98), 1)


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

    elif name in (
        "330 OHM",
        "1 kOHM",
        "2.2 kOHM",
        "3.3 kOHM",
        "10 kOHM",
        "100 kOHM"
    ):

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
    # DIODE
    # ========================================================

    elif name == "DIODE":

        color = (180, 80, 180)

        # Left terminal
        cv2.line(
            board,
            (x + 25, y + 45),
            (x + 65, y + 45),
            color,
            3
        )

        # Diode triangle/body
        points = np.array(
            [
                [x + 65, y + 20],
                [x + 65, y + 70],
                [x + 105, y + 45]
            ],
            np.int32
        )

        cv2.fillPoly(
            board,
            [points],
            color
        )

        # Cathode bar
        cv2.line(
            board,
            (x + 105, y + 18),
            (x + 105, y + 72),
            (80, 40, 80),
            4
        )

        # Right terminal
        cv2.line(
            board,
            (x + 105, y + 45),
            (x + 145, y + 45),
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
    # CURRENT SOURCE
    # ========================================================

    elif name == "CURRENT SOURCE":

        color = (180, 70, 70)

        cv2.circle(
            board,
            (x + 90, y + 45),
            25,
            color,
            3
        )

        cv2.arrowedLine(
            board,
            (x + 90, y + 58),
            (x + 90, y + 32),
            color,
            3,
            tipLength=0.25
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
            (x + 115, y + 45),
            (x + 155, y + 45),
            color,
            3
        )


    # ========================================================
    # NAME
    # ========================================================

    cv2.putText(
        board,
        name,
        (x + 10, y + 82),
        cv2.FONT_HERSHEY_PLAIN,
        1.15,
        (55, 55, 55),
        1
    )


# ============================================================
# LIBRARY COMPONENT ICONS
# Physical-component-inspired vector icons.
# ============================================================

def draw_library_icon(board, name, cx, cy, color):

    cx = int(cx)
    cy = int(cy)
    lead = (105, 100, 94)

    if name == "BATTERY":
        cv2.line(board, (cx - 30, cy), (cx - 9, cy), lead, 2, cv2.LINE_AA)
        cv2.line(board, (cx + 9, cy), (cx + 30, cy), lead, 2, cv2.LINE_AA)
        cv2.line(board, (cx - 6, cy - 10), (cx - 6, cy + 10), color, 2, cv2.LINE_AA)
        cv2.line(board, (cx + 6, cy - 15), (cx + 6, cy + 15), color, 2, cv2.LINE_AA)
        return

    if name in {"330 OHM", "1 kOHM", "2.2 kOHM",
                "3.3 kOHM", "10 kOHM", "100 kOHM"}:
        cv2.line(board, (cx - 30, cy), (cx - 19, cy), lead, 2, cv2.LINE_AA)
        cv2.line(board, (cx + 19, cy), (cx + 30, cy), lead, 2, cv2.LINE_AA)
        cv2.rectangle(board, (cx - 19, cy - 9), (cx + 19, cy + 9),
                      (218, 181, 125), -1)
        cv2.rectangle(board, (cx - 19, cy - 9), (cx + 19, cy + 9),
                      (170, 140, 95), 1)
        for xoff, band_color in [
            (-11, (45, 45, 45)),
            (-3, (180, 55, 40)),
            (5, (225, 175, 35)),
            (13, (55, 55, 55))
        ]:
            cv2.rectangle(board,
                          (cx + xoff, cy - 9),
                          (cx + xoff + 3, cy + 9),
                          band_color, -1)
        return

    if name == "LED":
        cv2.line(board, (cx - 30, cy), (cx - 12, cy), lead, 2, cv2.LINE_AA)
        cv2.line(board, (cx + 12, cy), (cx + 30, cy), lead, 2, cv2.LINE_AA)
        cv2.circle(board, (cx, cy), 10, (245, 245, 235), -1)
        cv2.circle(board, (cx, cy), 10, color, 2, cv2.LINE_AA)
        cv2.line(board, (cx - 7, cy - 14), (cx - 13, cy - 21), color, 1, cv2.LINE_AA)
        cv2.line(board, (cx + 7, cy - 14), (cx + 13, cy - 21), color, 1, cv2.LINE_AA)
        return

    if name == "CAPACITOR":
        cv2.line(board, (cx - 30, cy), (cx - 8, cy), lead, 2, cv2.LINE_AA)
        cv2.line(board, (cx + 8, cy), (cx + 30, cy), lead, 2, cv2.LINE_AA)
        cv2.line(board, (cx - 7, cy - 14), (cx - 7, cy + 14), color, 2, cv2.LINE_AA)
        cv2.line(board, (cx + 7, cy - 14), (cx + 7, cy + 14), color, 2, cv2.LINE_AA)
        return

    if name == "DIODE":
        cv2.line(board, (cx - 30, cy), (cx - 13, cy), lead, 2, cv2.LINE_AA)
        cv2.line(board, (cx + 9, cy), (cx + 30, cy), lead, 2, cv2.LINE_AA)
        points = np.array([
            [cx - 13, cy - 11],
            [cx - 13, cy + 11],
            [cx + 7, cy]
        ], np.int32)
        cv2.fillPoly(board, [points], color)
        cv2.line(board, (cx + 9, cy - 14), (cx + 9, cy + 14),
                 (75, 65, 65), 2, cv2.LINE_AA)
        return

    if name == "CURRENT SOURCE":
        cv2.line(board, (cx - 30, cy), (cx - 14, cy), lead, 2, cv2.LINE_AA)
        cv2.line(board, (cx + 14, cy), (cx + 30, cy), lead, 2, cv2.LINE_AA)
        cv2.circle(board, (cx, cy), 12, (248, 248, 248), -1)
        cv2.circle(board, (cx, cy), 12, color, 2, cv2.LINE_AA)
        cv2.arrowedLine(board, (cx, cy + 7), (cx, cy - 7),
                        color, 2, tipLength=0.30)
        return


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
drag_history_saved = False

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
# INDEX-FINGER WIRING
# ============================================================
# Wiring uses only the index fingertip.
# Hold on terminal 1 -> select.
# Move to terminal 2 and hold -> create wire.

INDEX_WIRE_HOLD_FRAMES = 15
index_wire_hold_frames = 0
index_wire_candidate = None

# Undo button
UNDO_X1 = 820
UNDO_Y1 = 20
UNDO_X2 = 950
UNDO_Y2 = 65


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
        board = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 242

        # Header
        cv2.rectangle(board, (0, 0), (WIDTH, 92), (24, 28, 36), -1)
        cv2.putText(board, "CIRCUIT LAB", (30, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 247, 250), 2)
        cv2.putText(board, "AI GESTURE CIRCUIT BUILDER", (31, 67),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (165, 174, 188), 1)

        # Header status
        cv2.rectangle(board, (660, 18), (805, 65), (39, 45, 55), -1)
        cv2.rectangle(board, (660, 18), (805, 65), (92, 102, 118), 1)
        cv2.putText(board, "●  LIVE", (681, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (110, 220, 145), 2)

        # Undo button - same coordinates used by the existing gesture logic
        cv2.rectangle(board, (UNDO_X1, UNDO_Y1), (UNDO_X2, UNDO_Y2), (48, 54, 64), -1)
        cv2.rectangle(board, (UNDO_X1, UNDO_Y1), (UNDO_X2, UNDO_Y2), (110, 120, 135), 1)
        cv2.putText(board, "UNDO", (UNDO_X1 + 28, UNDO_Y1 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 243, 247), 2)

        # Main workspace
        cv2.rectangle(board, (25, 112), (970, 615), (255, 255, 255), -1)
        cv2.rectangle(board, (25, 112), (970, 615), (214, 219, 227), 2)
        cv2.putText(board, "CIRCUIT CANVAS", (45, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (105, 113, 126), 1)

        # Subtle placement grid. It is visual only; component coordinates/logic are unchanged.
        for gx in range(50, 960, 40):
            cv2.line(board, (gx, 160), (gx, 600), (239, 242, 246), 1)
        for gy in range(175, 600, 40):
            cv2.line(board, (35, gy), (960, gy), (239, 242, 246), 1)

        # Component library
        cv2.rectangle(board, (985, 112), (1180, 615), (250, 251, 253), -1)
        cv2.rectangle(board, (985, 112), (1180, 615), (214, 219, 227), 2)
        cv2.putText(board, "COMPONENT LIBRARY", (1000, 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.39, (70, 77, 88), 1)
        cv2.putText(board, "Hover to select", (1000, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (145, 151, 161), 1)

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


            # IMPORTANT:
            # previous_pinch is updated AFTER the gesture block.
            # This preserves the original pinch-start behavior.


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
            # INDEX-FINGER-ONLY WIRING
            # =================================================
            #
            # This does NOT alter the original pinch gesture.
            # Pinch remains available for moving components and
            # the existing interaction remains otherwise unchanged.
            #
            # Wiring:
            #   1. Point index finger at terminal 1 and hold.
            #   2. Move to terminal 2 and hold.
            #   3. Wire is created.
            #
            # The index fingertip is already used by the original
            # cursor, so no new hand landmark detection is needed.

            index_component, index_terminal = find_terminal(
                cursor_x,
                cursor_y
            )

            if index_component is not None:

                candidate = (
                    index_component,
                    index_terminal
                )

                if candidate == index_wire_candidate:
                    index_wire_hold_frames += 1
                else:
                    index_wire_candidate = candidate
                    index_wire_hold_frames = 1

                if (
                    index_wire_hold_frames >= INDEX_WIRE_HOLD_FRAMES
                    and
                    cooldown == 0
                ):

                    if wire_start is None:

                        wire_start = index_component
                        wire_start_terminal = index_terminal

                        print(
                            "WIRE START:",
                            index_component["name"],
                            "TERMINAL:",
                            index_terminal
                        )

                    elif (
                        index_component is not wire_start
                        or
                        index_terminal != wire_start_terminal
                    ):

                        exists = False

                        for wire in wires:

                            same_direction = (
                                wire[0] is wire_start
                                and
                                wire[1] is index_component
                                and
                                wire[2] == wire_start_terminal
                                and
                                wire[3] == index_terminal
                            )

                            reverse_direction = (
                                wire[0] is index_component
                                and
                                wire[1] is wire_start
                                and
                                wire[2] == index_terminal
                                and
                                wire[3] == wire_start_terminal
                            )

                            if same_direction or reverse_direction:
                                exists = True
                                break

                        if not exists:

                            save_state()

                            wires.append(
                                (
                                    wire_start,
                                    index_component,
                                    wire_start_terminal,
                                    index_terminal
                                )
                            )

                            print(
                                "WIRE CREATED:",
                                wire_start["name"],
                                "TERMINAL",
                                wire_start_terminal,
                                "->",
                                index_component["name"],
                                "TERMINAL",
                                index_terminal
                            )

                        else:
                            print("WIRE ALREADY EXISTS")

                        wire_start = None
                        wire_start_terminal = None

                    # Prevent one long hold from creating repeated wires.
                    index_wire_hold_frames = 0
                    index_wire_candidate = None
                    cooldown = 15

            else:

                index_wire_hold_frames = 0
                index_wire_candidate = None


            # =================================================
            # PINCH / DRAGGING
            # =================================================

            pinch_started = is_pinching and not previous_pinch

            if pinch_started:

                # BACK / UNDO button uses the same pinch gesture.
                if (
                    UNDO_X1 <= cursor_x <= UNDO_X2
                    and
                    UNDO_Y1 <= cursor_y <= UNDO_Y2
                ):

                    undo_action()
                    cooldown = 20

                    pinch_start_x = None
                    pinch_start_y = None
                    dragging = False
                    dragged_component = None

                else:

                    pinch_start_x = cursor_x
                    pinch_start_y = cursor_y

                    terminal_component, terminal_index = find_terminal(
                        cursor_x,
                        cursor_y
                    )

                    if terminal_component is not None:

                        dragging = False
                        dragged_component = None

                        if wire_start is None:

                            wire_start = terminal_component
                            wire_start_terminal = terminal_index

                            print(
                                "WIRE START:",
                                terminal_component["name"],
                                "TERMINAL:",
                                terminal_index
                            )

                        else:

                            second_component = terminal_component
                            second_terminal = terminal_index

                            if (
                                second_component is not wire_start
                                or
                                second_terminal != wire_start_terminal
                            ):

                                exists = False

                                for wire in wires:

                                    same_direction = (
                                        wire[0] is wire_start
                                        and
                                        wire[1] is second_component
                                        and
                                        wire[2] == wire_start_terminal
                                        and
                                        wire[3] == second_terminal
                                    )

                                    reverse_direction = (
                                        wire[0] is second_component
                                        and
                                        wire[1] is wire_start
                                        and
                                        wire[2] == second_terminal
                                        and
                                        wire[3] == wire_start_terminal
                                    )

                                    if same_direction or reverse_direction:
                                        exists = True
                                        break

                                if not exists:

                                    save_state()

                                    wires.append(
                                        (
                                            wire_start,
                                            second_component,
                                            wire_start_terminal,
                                            second_terminal
                                        )
                                    )

                                    print(
                                        "WIRE CREATED:",
                                        wire_start["name"],
                                        "TERMINAL",
                                        wire_start_terminal,
                                        "->",
                                        second_component["name"],
                                        "TERMINAL",
                                        second_terminal
                                    )

                                else:
                                    print("WIRE ALREADY EXISTS")

                            wire_start = None
                            wire_start_terminal = None

                    else:

                        component = find_placed_component(
                            cursor_x,
                            cursor_y
                        )

                        if component is not None:

                            save_state()
                            drag_history_saved = True

                            dragging = True
                            dragged_component = component

                            grab_offset_x = cursor_x - component["x"]
                            grab_offset_y = cursor_y - component["y"]

            if is_pinching and dragging and dragged_component is not None:

                new_x = cursor_x - grab_offset_x
                new_y = cursor_y - grab_offset_y

                new_x = max(35, min(800, new_x))
                new_y = max(155, min(525, new_y))

                dragged_component["x"] = new_x
                dragged_component["y"] = new_y

            if not is_pinching:

                dragging = False
                dragged_component = None
                drag_history_saved = False
                pinch_start_x = None
                pinch_start_y = None

            previous_pinch = is_pinching

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

                    column = number % 3
                    row = number // 3

                    x = 70 + column * 280
                    y = 180 + row * 105

                    if y <= 510:

                        save_state()

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
            drag_history_saved = False

            pinch_start_x = None
            pinch_start_y = None
            previous_pinch = False


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
        # ====================================================
        # POLISHED STATUS / MENU
        # ====================================================

        # Circuit status pill
        if circuit_complete:
            status_label = "●  CIRCUIT COMPLETE"
            status_color = (80, 190, 120)
        else:
            status_label = "●  INCOMPLETE"
            status_color = (90, 130, 225)

        cv2.rectangle(board, (430, 18), (645, 65), (39, 45, 55), -1)
        cv2.rectangle(board, (430, 18), (645, 65), status_color, 1)
        cv2.putText(board, status_label, (446, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 2)

        # Menu cards: preserve the original menu coordinates so placement gestures
        # remain exactly the same.
        for name, x1, y1, x2, y2 in menu:
            selected_menu = (name == current_menu_component)
            if selected_menu:
                background = (225, 241, 255)
                border = (65, 145, 230)
                thickness = 3
            else:
                background = (255, 255, 255)
                border = (218, 222, 229)
                thickness = 1

            cv2.rectangle(board, (x1, y1), (x2, y2), background, -1)
            cv2.rectangle(board, (x1, y1), (x2, y2), border, thickness)

            # Real component-style icon + thin label.
            icon_x = x2 - 34
            icon_y = (y1 + y2) // 2
            icon_color = (92, 82, 72) if not selected_menu else (45, 125, 205)

            draw_library_icon(
                board,
                name,
                icon_x,
                icon_y,
                icon_color
            )

            cv2.putText(
                board,
                name,
                (x1 + 14, y1 + 29),
                cv2.FONT_HERSHEY_PLAIN,
                1.15,
                (55, 60, 68),
                1,
                cv2.LINE_AA
            )

        # ====================================================
        # DRAW WIRES
        # ====================================================

        for wire in wires:

            draw_wire(
                board,
                wire[0],
                wire[2],
                wire[1],
                wire[3]
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
        # CURSOR + INTERACTION FEEDBACK
        # ====================================================

        if cursor_x is not None:
            if is_pinching:
                cursor_color = (205, 80, 210)
            elif wire_start is not None:
                cursor_color = (70, 185, 110)
            else:
                cursor_color = (70, 125, 225)

            cv2.circle(board, (cursor_x, cursor_y), 18, cursor_color, 2)
            cv2.circle(board, (cursor_x, cursor_y), 4, cursor_color, -1)

        # Interaction card
        card_x1, card_y1, card_x2, card_y2 = 35, 535, 950, 595
        cv2.rectangle(board, (card_x1, card_y1), (card_x2, card_y2), (247, 249, 252), -1)
        cv2.rectangle(board, (card_x1, card_y1), (card_x2, card_y2), (224, 228, 235), 1)

        if dragging and dragged_component is not None:
            state_title = "MOVING"
            state_text = "Release the pinch to place " + dragged_component["name"]
            state_color = (225, 145, 65)
        elif wire_start is not None:
            state_title = "CONNECTING"
            state_text = ("Index finger: move to another terminal — "
                          + wire_start["name"] + " T" + str(wire_start_terminal + 1))
            state_color = (55, 175, 105)
        elif current_menu_component is not None:
            seconds = stable_frames / 30
            state_title = "PLACING"
            state_text = ("Hold " + current_menu_component + "  " +
                          str(round(seconds, 1)) + " / 0.7 sec")
            state_color = (70, 125, 225)
        else:
            state_title = "READY"
            state_text = "Use your existing gestures to place, move and connect components"
            state_color = (105, 113, 126)

        cv2.putText(board, state_title, (55, 558),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, state_color, 1)
        cv2.putText(board, state_text, (155, 558),
                    cv2.FONT_HERSHEY_PLAIN, 1.20, (58, 65, 76), 1)

        # ====================================================
        # FOOTER
        # ====================================================

        cv2.rectangle(board, (0, 630), (WIDTH, 700), (24, 28, 36), -1)
        cv2.putText(board, "STATUS", (28, 658),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (145, 155, 170), 1)
        cv2.putText(board, circuit_message, (100, 658),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, (242, 245, 248), 1)

        footer_status = "COMPLETE" if circuit_complete else "BUILDING"
        footer_color = (100, 215, 145) if circuit_complete else (225, 170, 80)
        cv2.putText(board, footer_status, (1060, 658),
                    cv2.FONT_HERSHEY_PLAIN, 1.25, footer_color, 1)

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

        if key == ord("z") or key == ord("Z") or key == 8:
            undo_action()


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Circuit Vision closed.")

