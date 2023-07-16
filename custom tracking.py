from cv2 import cv2
import numpy as np
import math
from random import randint
from lib.Box import Box
cv = cv2

# Resize binary, threshold img height to this amount, maintain 16:9 ratio
SCALE_HEIGHT = 144  # 96x54 or 256x144 both seems reasonable
# Multiply by this factor to convert 720p scale to the current scale. Divide to undo
CONVERSION_720P = SCALE_HEIGHT / 720

cap = cv2.VideoCapture(r"C:\Users\Amy\Documents\python\DealNo Solver\IMG_4383.MOV")
cap.set(cv2.CAP_PROP_POS_FRAMES, 900)


def hsv2bgr(h, s, v):
    # Don't question it
    return tuple(map(int, cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]))


class Case(Box):
    TOLERANCE_X = int(30 * CONVERSION_720P)
    TOLERANCE_Y = int(20 * CONVERSION_720P)

    def __init__(self, xywh: tuple, value: int = None, momentum: tuple[float, float] = None, color: tuple[int, int, int] = None):
        self.value = value
        self.momentum = (0, 0)
        super().__init__(xywh=xywh)
        # Must be re-set after super init b/c constructor calls set_pos, which is overridden by Case and updates momentum
        self.momentum = momentum or (0, 0)

        # Debug features
        self.color = color or hsv2bgr(randint(0, 255), 255, 255)
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
        if big_self.chk_collision(bound) and (bound.h > 25 * CONVERSION_720P or bound.w > 55 * CONVERSION_720P):
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
        step_size = int(10 * CONVERSION_720P)
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
                    (self.x1, self.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        cx, cy = self.center
        mx, my = self.momentum
        cv2.line(frame, (cx, cy), (int(cx + mx), int(cy + my)), self.color, 2)
        super().show(cv2, frame, self.color)
        # cv.rectangle(frame, (case.x1, case.y1), (case.x2, case.y2), case.color, 2)


def _getBox(bin_frame: np.ndarray, centroid: Box):
    """"inflate" from center until hit wall (starts as square but can become rectangle)"""
    # Cool related algorithms (not implemented):
    # Maximize the rectangular area under Histogram: https://stackoverflow.com/a/35931960
    # Find largest rectangle containing only zeros in an N×N binary matrix: https://stackoverflow.com/a/12387148
    cx, cy = centroid.center
    x1 = x2 = cx
    y1 = y2 = cy

    # Stop iterating when all 4 edges cannot expand further
    while True:
        x1 -= 1
        y1 -= 1
        x2 += 1
        y2 += 1
        (Box(xyxy=(x1, y1, x2, y2)) / CONVERSION_720P).show(cv2, frame=frame, color=(0, 0, 255))
        illegalEdges = 0
        # Adding 1 to range upper bound Solves issue where boxes centered in black infinitely stretch
        # First two checks (top & bottom) fail, reducing vertical search to [], which never realizes the other 2 edges are illegal
        # Check top edge
        for x in range(x1, x2 + 1):
            if bin_frame[y1, x] == 0:
                illegalEdges += 1
                y1 += 1
                break
        # Check bottom edge
        for x in range(x1, x2 + 1):
            if bin_frame[y2, x] == 0:
                illegalEdges += 1
                y2 -= 1
                break
        # Check left edge
        for y in range(y1, y2 + 1):
            if bin_frame[y, x1] == 0:
                illegalEdges += 1
                x1 += 1
                break
        # Check right edge
        for y in range(y1, y2 + 1):
            if bin_frame[y, x2] == 0:
                illegalEdges += 1
                x2 -= 1
                break
        if illegalEdges == 4:
            newBox = Box(xyxy=(x1, y1, x2, y2))
            # This is the case where boxes have swapped & are waiting to separate & continue tracking. Do not update centroid if it is not legal shape
            # By some quirk of the algorithm, centroids surrounded by black are 2 by -2
            if centroid.area <= 0 and (newBox.h > newBox.w or newBox.w > newBox.h * 2):
                return centroid
            return newBox

def move_swap_boxes(frame: np.ndarray, centroids: list[Case]):
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


def tick() -> bool:
    """Take a frame and update case state
    :return: boolean of whether mainloop should continue
    """
    global frame
    ret, frame = cap.read()
    if not ret:
        return False
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bin", binary)  # For reference
    kernel_size_720p = 60  # 80x80 does not work b/c prev centroid outside case after moving
    kernel_size = int(kernel_size_720p * (frame.shape[0] / 720))  # convert kernel size for 720p to current resolution
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Ref: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    erosion = cv.erode(binary, kernel)

    # Scale to remove unnecessary info (assumes 9:16 ratio)
    erosion = cv2.resize(erosion, (int(SCALE_HEIGHT * (16/9)), SCALE_HEIGHT))

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

            if len(contours) == 16:  # First time init
                case = Case((x, y, w, h), randint(1, 99))
                cases.append(case)
                case.set_pos(box=_getBox(erosion, case))
        if len(contours) == 16:  # First time init
            # Tried scaling boxes by * .5 and .75, but no improvement (worse even)
            avg_width = round(sum(map(lambda c: c.w, cases)) / 16)
            avg_height = round(sum(map(lambda c: c.h, cases)) / 16)
            for case in cases:
                case.size = (avg_width, avg_height)
                case.momentum = (0, 0)
    else:
        # Coordinates are used as (x, y), but defined as (y, x). I think it is consistent & no issues
        HERE = np.array((0, 0))
        UP = np.array((-1, 0))
        DOWN = np.array((1, 0))
        LEFT = np.array((0, -1))
        RIGHT = np.array((0, 1))
        DIRECTIONS = (HERE, UP, DOWN, LEFT, RIGHT)
        for case in cases:
            original_position = np.array(case.center)
            original_coverage = erosion[case.slicer].sum()
            best_direction = HERE
            best_coverage = 0
            for check_distance in range(1, 100):
                for direction in DIRECTIONS:
                    case.moveBy(*(direction * check_distance))
                    new_sum = erosion[case.slicer].sum()
                    if new_sum > best_coverage:
                        best_coverage = new_sum
                        best_direction = direction
                    case.moveBy(*(direction * check_distance * -1))
                if best_coverage > original_coverage or original_coverage > 0:
                    case.moveBy(*(best_direction * check_distance))
                    break

            if best_direction is HERE:
                case.moveBy(*case.momentum)

            best_coverage = 0
            while True:
                coverage = erosion[case.slicer].sum()
                case.moveBy(*best_direction)
                if coverage <= best_coverage:
                    case.moveBy(*(best_direction * -2))  # TODO: no clue why * -2, was expecting -1
                    break
                best_coverage = coverage

            case.momentum = (case.center - original_position + case.momentum) / 2 if original_coverage != 0 else HERE

    cv2.imshow('smol', erosion)
    # Display case values
    erosion = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)
    # Scale back up for ease of debugging
    erosion = cv2.resize(erosion, (1280, 720), interpolation=cv.INTER_NEAREST)
    for case in cases:
        Case((case / CONVERSION_720P).xywh, case.value, (case.momentum[0] / CONVERSION_720P, case.momentum[1] / CONVERSION_720P), case.color).show(frame=erosion)

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
        return False
    return True


cases: list[Case] = []
frame: np.ndarray = None  # Global scope OK since only used for debugging
if __name__ == "__main__":
    while tick():
        # No-op: tick() has all logic
        pass
    cap.release()
    cv2.destroyAllWindows()

