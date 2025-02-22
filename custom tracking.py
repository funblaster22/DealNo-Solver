import cv2
from time import time
import numpy as np
from scipy import stats
from collections import deque
from random import randint
from concurrent.futures import ThreadPoolExecutor
from lib.Box import Box
cv = cv2

# Resize binary, threshold img height to this amount, maintain 16:9 ratio
SCALE_HEIGHT = 144  # 96x54 or 256x144 both seems reasonable
# Multiply by this factor to convert 720p scale to the current scale. Divide to undo
CONVERSION_720P = SCALE_HEIGHT / 720


def hsv2bgr(h, s, v):
    # Don't question it
    return tuple(map(int, cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2BGR)[0, 0]))


class Case(Box):
    def __init__(self, xywh: tuple, value: int = None, momentum: tuple[float, float] = None, color: tuple[int, int, int] = None):
        self.value = value
        self.momentum = (0, 0)
        self.momentum_history = deque(maxlen=3)
        # How many frames to abstain from moving (set after swapping)
        self.cooldown = 0
        super().__init__(xywh=xywh)
        # Must be re-set after super init b/c constructor calls set_pos, which is overridden by Case and updates momentum
        self.momentum = momentum or (0, 0)

        # Debug features
        self.color = color or hsv2bgr(randint(0, 255), 255, 255)

    def project(self, bin_frame: np.ndarray):
        """pr-oh-ject: keep moving Box in direction of previous velocity until leaving white region.
        This works because while swapping, the combined bounding box should decrease"""
        velocityX, velocityY = stats.mode(self.momentum_history).mode.astype(int)
        assert not (velocityX == 0 and velocityY == 0)
        # TODO: if velocity is 0, invert the momentum of the opposing case
        assert velocityX != velocityY
        cx, cy = self.center
        while bin_frame[cy, cx] != 0:
            cx += velocityX
            cy += velocityY
        self.center = (cx + velocityX * int(self.w / 2 + 3), cy + velocityY * int(self.h / 2 + 3))
        self.cooldown = 12
        self.momentum_history.clear()

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


def cardinal_unit_vector(vector):
    return (np.sign(vector[0]) if abs(vector[0]) > abs(vector[1]) else 0,
            np.sign(vector[1]) if abs(vector[1]) > abs(vector[0]) else 0)


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
        (Box(xyxy=(x1, y1, x2, y2)) / CONVERSION_720P).show(cv2, frame=debug_frame, color=(0, 0, 255))
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


def preprocess_frame(frame: np.ndarray):
    """Convert to binary, erode to make separation clearer, and scale down to `SCALE_HEIGHT` pixels tall"""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grey, 100, 255, cv2.THRESH_BINARY)
    # cv2.imshow("bin", binary)  # For reference
    kernel_size_720p = 60  # 80x80 does not work b/c prev centroid outside case after moving
    kernel_size = int(kernel_size_720p * (frame.shape[0] / 720))  # convert kernel size for 720p to current resolution
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Ref: https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    erosion = cv.erode(binary, kernel)

    # Scale to remove unnecessary info (assumes 9:16 ratio)
    return cv2.resize(erosion, (int(SCALE_HEIGHT * (16 / 9)), SCALE_HEIGHT))


def iter_cap(vid_src: str):
    """Iterate through every other frame of a video"""
    cap = cv2.VideoCapture(vid_src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 900)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 2 == 1:
            continue
        yield frame
    cap.release()


def preprocess(vid_src: str):
    """Asynchronously iterate through frames and apply `preprocess_frame`
    Rationale: Since pre-processing does not depend on prior frames, it is a prime candidate for parallelization
    :returns: array with applied transformations
    """
    # Unfortunately, multiprocessing is slower (Tested multiprocessing.Pool.map & ProcessPoolExecutor w/ different chunksize/process)
    # return list(map(preprocess_frame, iter_cap(vid_src)))  # Synchronous
    with ThreadPoolExecutor() as executor:  # You can use ProcessPoolExecutor() for multiprocessing
        return executor.map(preprocess_frame, iter_cap(vid_src))


def tick(bin_frame: np.ndarray, debug: bool) -> bool:
    """Take a frame and update case state
    :return: boolean of whether mainloop should continue
    """
    global debug_frame
    debug_frame = cv2.cvtColor(bin_frame, cv2.COLOR_GRAY2BGR)
    # Scale back up for ease of debugging
    debug_frame = cv2.resize(debug_frame, (1280, 720), interpolation=cv.INTER_NEAREST)

    # TODO: first attempt may not be correct, so check over several iters
    if len(cases) < 16:  # First time init
        contours, _hierarchy = cv.findContours(bin_frame, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            # Cases are always wider than tall. Disqualify batch if violated
            if h > w:
                cases.clear()
                break
            cv.circle(debug_frame, (int(x + w / 2), int(y + h / 2)), 2, (0, 0, 255), -1)

            if len(contours) == 16:  # First time init
                case = Case((x, y, w, h), randint(1, 99))
                cases.append(case)
                case.set_pos(box=_getBox(bin_frame, case))
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
            if case.cooldown > 0:
                case.cooldown -= 1
                continue
            original_position = np.array(case.center)
            original_coverage = bin_frame[case.slicer].sum()
            best_direction = HERE
            best_coverage = 0
            for direction in DIRECTIONS:
                case.moveBy(*direction)
                new_sum = bin_frame[case.slicer].sum()
                if new_sum > best_coverage:
                    best_coverage = new_sum
                    best_direction = direction
                case.moveBy(*(direction * -1))

            if best_direction is HERE and best_coverage > 0:
                case.moveBy(int(case.momentum[0]), int(case.momentum[1]))

            best_coverage = 0
            while True:
                coverage = bin_frame[case.slicer].sum()
                case.moveBy(*best_direction)
                if coverage <= best_coverage:
                    case.moveBy(*(best_direction * -2))  # TODO: no clue why * -2, was expecting -1
                    break
                best_coverage = coverage

            case.momentum = (case.center - original_position + case.momentum) / 2 if original_coverage != 0 else HERE
            case.momentum_history.append(cardinal_unit_vector(case.momentum))

            if len(case.momentum_history) == case.momentum_history.maxlen:
                for collision in cases:
                    if case != collision and Box.intersection(case, collision):
                        if stats.mode(case.momentum_history).count.min() > 1 and stats.mode(collision.momentum_history).count.min() > 1:
                            case.project(bin_frame)
                            collision.project(bin_frame)
                        break

    if debug:
        cv2.imshow('bin_frame', bin_frame)
        for case in cases:
            Case((case / CONVERSION_720P).xywh, case.value, (case.momentum[0] / CONVERSION_720P, case.momentum[1] / CONVERSION_720P), case.color).show(frame=debug_frame)

        cv2.imshow('debug_frame', debug_frame)
        if len(cases) < 16:
            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(1)
        print(".", end="")
        if key == ord('q'):
            return False
    return True


def main(debug: bool):
    # Reset global vars
    global cases, debug_frame
    cases = []
    debug_frame = None

    print("Start!")
    start = time()
    vid_src = "IMG_4383.MOV"
    bin_frames = preprocess(vid_src)
    preprocess_end = time()
    print("Preprocessed in:", round(preprocess_end - start, 2), "seconds")
    for frame in bin_frames:
        if not tick(frame, debug):
            break
    print("Tracked in:", round(time() - preprocess_end, 2), "seconds")
    print("Total time:", round(time() - start, 2), "seconds")
    cv2.destroyAllWindows()


cases: list[Case] = []
debug_frame: np.ndarray = None  # Global scope OK since only used for debugging
if __name__ == "__main__":
    debug = True  # Show preview or not
    main(debug)
