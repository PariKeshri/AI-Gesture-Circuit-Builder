import cv2
import numpy as np

# ============================================================
# CIRCUIT VISION - VIRTUAL CIRCUIT BOARD
# ============================================================

WIDTH = 1200
HEIGHT = 700

# White background
board = np.ones(
    (HEIGHT, WIDTH, 3),
    dtype=np.uint8
) * 255


# ============================================================
# TITLE
# ============================================================

cv2.putText(
    board,
    "CIRCUIT VISION",
    (30, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.1,
    (30, 30, 30),
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


# ============================================================
# CIRCUIT WORKSPACE
# ============================================================

cv2.rectangle(
    board,
    (30, 110),
    (850, 640),
    (210, 210, 210),
    2
)

cv2.putText(
    board,
    "CIRCUIT WORKSPACE",
    (50, 140),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (120, 120, 120),
    1
)


# ============================================================
# COMPONENT PANEL
# ============================================================

cv2.rectangle(
    board,
    (880, 110),
    (1170, 640),
    (240, 240, 240),
    -1
)

cv2.rectangle(
    board,
    (880, 110),
    (1170, 640),
    (180, 180, 180),
    2
)

cv2.putText(
    board,
    "COMPONENTS",
    (925, 150),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.65,
    (30, 30, 30),
    2
)


# ============================================================
# COMPONENT BUTTON FUNCTION
# ============================================================

def draw_component_button(
    image,
    name,
    y_position
):

    x1 = 915
    x2 = 1135

    cv2.rectangle(
        image,
        (x1, y_position),
        (x2, y_position + 60),
        (255, 255, 255),
        -1
    )

    cv2.rectangle(
        image,
        (x1, y_position),
        (x2, y_position + 60),
        (80, 80, 80),
        2
    )

    cv2.putText(
        image,
        name,
        (x1 + 25, y_position + 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (30, 30, 30),
        1
    )


# ============================================================
# COMPONENTS
# ============================================================

draw_component_button(
    board,
    "BATTERY",
    180
)

draw_component_button(
    board,
    "RESISTOR",
    270
)

draw_component_button(
    board,
    "LED",
    360
)

draw_component_button(
    board,
    "CAPACITOR",
    450
)


# ============================================================
# INSTRUCTIONS
# ============================================================

cv2.putText(
    board,
    "GESTURES",
    (60, 530),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (50, 50, 50),
    1
)

cv2.putText(
    board,
    "Point  = Select",
    (60, 560),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (80, 80, 80),
    1
)

cv2.putText(
    board,
    "Fist   = Grab / Move",
    (60, 585),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (80, 80, 80),
    1
)

cv2.putText(
    board,
    "2 Fingers = Rotate",
    (60, 610),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (80, 80, 80),
    1
)


# ============================================================
# STATUS
# ============================================================

cv2.rectangle(
    board,
    (30, 650),
    (1170, 690),
    (30, 30, 30),
    -1
)

cv2.putText(
    board,
    "Select a component to begin building your circuit",
    (300, 677),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.5,
    (255, 255, 255),
    1
)


# ============================================================
# SHOW WINDOW
# ============================================================

cv2.imshow(
    "Circuit Vision - Circuit Builder",
    board
)

print()
print("======================================")
print("       CIRCUIT VISION")
print("       CIRCUIT BUILDER")
print("======================================")
print()
print("Board opened successfully.")
print("Press Q to close.")
print()


# ============================================================
# WAIT
# ============================================================

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ============================================================
# CLOSE
# ============================================================

cv2.destroyAllWindows()

print("Circuit Vision closed.")