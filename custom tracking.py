from cv2 import cv2
import numpy as np
import math
from random import randint
from lib.Box import Box

cv = cv2

cap = cv2.VideoCapture(r"C:\Users\Amy\Documents\python\DealNo Solver\IMG_4383.MOV")
cap.set(cv2.CAP_PROP_POS_FRAMES, 900)


def hsv2bgr(h, s, v):
    # Don't question it
    return tuple(map(int, cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]))


class Case(Box):
    TOLERANCE_X = 30
    TOLERANCE_Y = 20

    def __init__(self, pos: tuple, value: int = None):
        self.value = value
        self.momentum = (0, 0)
        super().__init__(xywh=pos)

        # Debug features
        self.color = hsv2bgr(randint(0, 255), 255, 255)
        print(self.w * self.h, self.color)

    def recalc_pos(self, pos: Box):
        try:  # compute momentum
            self.momentum = (pos.cx - self.cx, pos.cy - self.cy)
        except AttributeError:
            pass
        self.set_pos()

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
                other.momentum = (0, 0)
                self.momentum = (0, 0)
            return True
        return False

    def project(self, bin_frame: np.ndarray):
        """pr-oh-ject: keep moving Box in direction of previous velocity until leaving white region.
        This works because while swapping, the combined bounding box should decrease"""
        speed = math.sqrt(self.momentum[0] ** 2 + self.momentum[1] ** 2)
        # if speed == 0:
        #     return
        velocityX = round(self.momentum[0] / speed)
        velocityY = round(self.momentum[1] / speed)
        assert velocityX != velocityY
        cx, cy = self.center
        while bin_frame[cy, cx] != 0:
            cx += velocityX
            cy += velocityY
        step_size = 10
        self.center = (cx + velocityX * step_size, cy + velocityY * step_size)

    def set_pos(self, **kwargs):
        cx1, cy1 = self.center
        cx2, cy2 = Box(**kwargs).center
        self.momentum = (
            ((cx2 - cx1) + self.momentum[0]) / 2,
            ((cy2 - cy1)  + self.momentum[1]) / 2
        )
        super().set_pos(**kwargs)

    def show(self, module=None, frame=None, color=None):
        """`module` and `color` are unused"""
        cv2.putText(frame, str(self.value),  # + ':' + str(max(abs(case.momentum[0]), abs(case.momentum[1]))),
                    (self.x1, self.y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        cx, cy = self.center
        mx, my = self.momentum
        cv2.line(frame, (cx, cy), (int(cx + mx), int(cy + my)), self.color, 2)
        super().show(cv2, frame, self.color)
        # cv.rectangle(frame, (case.x1, case.y1), (case.x2, case.y2), case.color, 2)


def _getBox(frame: np.ndarray, centroid: Box):
    """"inflate" from center until hit wall (starts as square but can become rectangle)"""
    # Cool related algorithms (not implemented):
    # Maximize the rectangular area under Histogram: https://stackoverflow.com/a/35931960
    # Find largest rectangle containing only zeros in an N×N binary matrix: https://stackoverflow.com/a/12387148
    cx, cy = centroid.center
    x1 = x2 = cx
    y1 = y2 = cy
    # if frame[cy, cx] == 0:
    #     raise LookupError("There is no box where the last centroid was!")

    # Stop iterating when all 4 edges cannot expand further
    while True:
        x1 -= 1
        y1 -= 1
        x2 += 1
        y2 += 1
        illegalEdges = 0
        # Check top edge
        for x in range(x1, x2):
            if (frame[y1, x]) == 0:
                illegalEdges += 1
                y1 += 1
        # Check bottom edge
        for x in range(x1, x2):
            if (frame[y2, x]) == 0:
                illegalEdges += 1
                y2 -= 1
        # Check left edge
        for y in range(y1, y2):
            if (frame[y, x1]) == 0:
                illegalEdges += 1
                x1 += 1
        # Check right edge
        for y in range(y1, y2):
            if (frame[y, x2]) == 0:
                illegalEdges += 1
                x2 -= 1
        if illegalEdges == 4:
            return Box(xyxy=(x1, y1, x2, y2))
        # TODO: can be shortened? (doesn't work for some reason)
        """
        illegalEdges = 0
        x1 -= 1
        x2 += 1
        for x in range(x1, x2):
            # Check top edge
            if (frame[y1, x]) == 0:
                illegalEdges += 1
                y1 += 1
            # Check bottom edge
            if (frame[y2, x]) == 0:
                illegalEdges += 1
                y2 -= 1
        y1 -= 1
        y2 += 1
        for y in range(y1, y2):
            # Check left edge
            if (frame[y, x1]) == 0:
                illegalEdges += 1
                x1 += 1
            # Check right edge
            if (frame[y, x2]) == 0:
                illegalEdges += 1
                x2 -= 1
        if illegalEdges == 4:
            return Box(xyxy=(x1, y1, x2, y2))
        """

def getBoxes(frame: np.ndarray, centroids: list[Case]):
    """See _getBox. Does that for every centroid, then checks collisions & swaps them"""
    # TODO: this is slow. Improve by 1: implementing natively (hard) or 2: downscaling img to 40x40?
    for centroid in centroids:
        newBox = _getBox(frame, centroid)
        centroid.set_pos(box=newBox)
        # check for overlap & swap
        for otherCentroid in centroids:
            if Box.intersection(centroid, otherCentroid) and centroid is not otherCentroid:
                centroid.project(frame)
                otherCentroid.project(frame)
    return centroids


cases: list[Case] = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bin", binary)  # For reference
    kernel = np.ones((60, 60), np.uint8)  # 80x80 does not work b/c prev centroid outside case after moving
    # Ref: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    erosion = cv.erode(binary, kernel)

    # TODO: first attempt may not be correct, so check over several iters
    if len(cases) < 16:  # First time init
        contours, _hierarchy = cv.findContours(erosion, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            # Cases are always wider than tall. Disqualify batch if violated
            if h > w:
                cases.clear()
                break
            box = Box(xywh=(x, y, w, h))
            cv.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (0, 0, 255), -1)

            # swapWith = None
            # for case in cases:  # test each case against new cases to move
            #     if case.try_swap(box, swapWith):
            #         swapWith = case
            #     else:
            #         case.try_move(box)

            if len(contours) == 16:  # First time init
                cases.append(Case((x, y, w, h), randint(1, 99)))
    else:
        # Use prev. centroid position to calculate next
        for box in getBoxes(erosion, cases):
            x, y, w, h = box.xywh
            cv.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (0, 0, 255), -1)
            box.show(cv2)

    # Display case values
    erosion = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)
    for case in cases:
        case.show(frame=frame)
        case.show(frame=erosion)

    cv2.imshow('contours', frame)
    cv2.imshow('thresh', erosion)
    if len(cases) < 16:
        key = cv2.waitKey(1)
    else:
        key = cv2.waitKey(0)
    # TEMP for my sanity TODO: remove
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == 930:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 962)
    print(".", end="")
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
