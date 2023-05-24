from cv2 import cv2
import numpy as np
from random import randint
from lib.Box import Box

cv = cv2

cap = cv2.VideoCapture(r"C:\Users\Amy\Documents\python\DealNo Solver\IMG_4383.MOV")


def hsv2bgr(h, s, v):
    # Don't question it
    return tuple(map(int, cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]))


class Case(Box):
    TOLERANCE_X = 30
    TOLERANCE_Y = 20

    def __init__(self, pos: tuple, value: int = None):
        super().__init__(xywh=pos)
        self.recalc_pos(self)
        self.value = value
        self.momentum = [0, 0]

        # Debug features
        self.color = hsv2bgr(randint(0, 255), 255, 255)
        print(self.w * self.h, self.color)

    def recalc_pos(self, pos: Box):
        try:  # compute momentum
            self.momentum[0] = pos.cx - self.cx
            self.momentum[1] = pos.cy - self.cy
        except AttributeError:
            pass
        self.set_pos(box=pos)

    def try_move(self, newPos: Box):
        dx, dy = newPos - self

        '''left = self.momentum[0] if self.momentum[0] < 0 else 0
        right = self.momentum[0] if self.momentum[0] > 0 else 0
        down = self.momentum[1] if self.momentum[1] < 0 else 0
        up = self.momentum[1] if self.momentum[1] > 0 else 0'''
        left = right = up = down = 0
        if left - self.TOLERANCE_X + right / 2 < dx < self.TOLERANCE_X + right + left / 2 and down - self.TOLERANCE_Y + up / 2 < dy < self.TOLERANCE_Y + up + down / 2:  # and w * h < 150
            self.recalc_pos(newPos)
            cv.rectangle(frame, (
            int(self.cx + left - self.TOLERANCE_X + right / 2), int(self.cy + down - self.TOLERANCE_Y + up / 2)),
                         (int(self.cx + self.TOLERANCE_X + right + left / 2),
                          int(self.cy + self.TOLERANCE_Y + up + down / 2)), self.color, 2)
        '''else:
            self.momentum[0] *= 1.1
            self.momentum[1] *= 1.1'''

    def try_swap(self, bound: Box, other: "Case") -> bool:
        big_self = Box(ccwh=(self.cx, self.cy, self.w + self.TOLERANCE_X * 2, self.h + self.TOLERANCE_Y * 2))
        big_self.show(cv, frame)
        if big_self.chk_collision(bound) and (bound.h > 25 or bound.w > 55):
            print("COLLISION", big_self, bound, other)
            if other is not None:
                print("SWAP", self.value, other.value)
                newPos = Box(box=other)
                other.recalc_pos(self)
                self.recalc_pos(newPos)
                other.momentum = [0, 0]
                self.momentum = [0, 0]
            return True
        return False


def getBoxes(frame):
    """Iterate through all pixels. When reaching a white pixel, find bottom and right of containing rect. Repeat until midpoint stops changing"""
    for y1 in range(frame.shape[0]):
        for x1 in range(frame.shape[1]):
            oldBbox = Box()
            while True:  # Repeat until midpoints stay the same
                cell = frame[y1, x1]
                if cell == 0:
                    break
                y2 = y1
                while frame[y2, x1] != 1:
                    y2 += 1
                midY = (y1 + y1) >> 1  # Fast average
                x2 = x1
                # TODO: can still create skinny rectangles
                # Initial plan: scanner initially jumps >1, if black, cut vertical distance and check from 1/4 & 3/4 y marks
                # However, this is variable on case size
                while frame[midY, x2] != 1:
                    x2 += 1
                newBbox =  Box(xyxy=(x1, y1, x2, y2))
                if newBbox == oldBbox:
                    yield newBbox


cases = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bin", binary)  # For reference
    kernel = np.ones((80, 80), np.uint8)  # also works well at 60x60
    # Ref: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    erosion = cv.erode(binary, kernel)

    contours, _hierarchy = cv.findContours(erosion, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    boxes = getBoxes(erosion)
    # cv.drawContours(frame, contours, -1, (0, 255, 0), 3)
    for box in boxes:
        x, y, w, h = box.xywh
        cv.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (0, 0, 255), -1)

        swapWith = None
        for case in cases:  # test each case against new cases to move
            if case.try_swap(box, swapWith):
                swapWith = case
            else:
                case.try_move(box)

        if len(contours) == 16 and len(cases) < 16:  # First time init
            cases.append(Case((x, y, w, h), randint(1, 99)))

    # Display case values
    erosion = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)
    for case in cases:
        cv2.putText(frame, str(case.value) + ':' + str(max(abs(case.momentum[0]), abs(case.momentum[1]))),
                    (case.x1, case.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, case.color, 2)
        cv.rectangle(frame, (case.x1, case.y1), (case.x2, case.y2), case.color, 2)

    cv2.imshow('contours', frame)
    cv2.imshow('thresh', erosion)
    if len(cases) < 16:
        key = cv2.waitKey(1)
    else:
        key = cv2.waitKey(0)
    print()
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
