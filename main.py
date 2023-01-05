import cv2
import numpy as np


def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            y1 = y1 + 400
            y2 = y2 + 400
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane


def pixel_points(y1, y2, line):
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    if x2 > 1000 or x2 < 100:  # prevents horizontal lines that are obviously not lanes
        return None
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    if left_lane is None:
        left_line = None
    else:
        slope_left, intercept_left = left_lane
        if intercept_left < 600:  # prevents lines who will get out of wanted scope
            left_line = None
        else:
            left_line = pixel_points(y1, y2, left_lane)
    if right_lane is None:
        right_line = None
    else:
        slope_right, intercept_right = right_lane
        y_right = slope_right * image.shape[1] + intercept_right
        if y_right < 600:  # prevents lines who will get out of wanted scope
            right_line = None
        else:
            right_line = pixel_points(y1, y2, right_lane)
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (0, 0, 0)
    thickness = 2
    if left_line is not None:
        if 560 < left_line[0][0] < 600:
            cv2.putText(frame, "Movement towards left lane was detected.", org, font, fontScale, color, thickness,
                        cv2.LINE_AA)
    if right_line is not None:
        if 640 > right_line[0][0] > 600:
            cv2.putText(frame, "Movement towards right lane was detected.", org, font, fontScale, color, thickness, cv2.LINE_AA)
    return left_line, right_line


def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=4):
    for line in lines:
        if line is not None:
            image = cv2.line(image, (line[0][0], line[0][1]), (line[1][0], line[1][1]), color, thickness, cv2.LINE_AA)
    return image


video = cv2.VideoCapture(r'/Users/edenbenzimra/PycharmProjects/LaneDetection/dashcam_video.mp4')
if video.isOpened():
    width = video.get(3)  # float `width`
    height = video.get(4)  # float `height`
size = (int(width), int(height))
currentframe = 0
img_array = []
while True:
    ret, frame = video.read()
    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cropped_frame = frame_gray[400:][:]
        mask_white = cv2.inRange(cropped_frame, int(cropped_frame.min()) + 200, 255)

        th_low = 100
        th_high = 200
        mag_im = cv2.Canny(mask_white, th_low, th_high)

        height, width = mag_im.shape
        vertices = np.array([
            [(0, height), (530, 0), (690, 0), (width, height)]
        ])
        vertices_for_frame = np.array([
            [(0, frame.shape[0]), (530, 432), (690, 432), (width, frame.shape[0])]
        ])

        overlay = frame.copy()
        cv2.fillPoly(overlay, pts=[vertices_for_frame], color=(255, 255, 255))
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        mask = np.zeros_like(mag_im)
        mask = cv2.fillPoly(mask, vertices, 255)
        mask = cv2.bitwise_and(mag_im, mask)

        r_step = 3
        t_step = np.pi / 180
        TH = 15
        lines = cv2.HoughLinesP(mask, r_step, t_step, TH, np.array([]), minLineLength=110, maxLineGap=60)
        res = frame.copy()

        if lines is not None:
            res = draw_lane_lines(frame, lane_lines(frame, lines))

        img_array.append(res)

    else:
        break
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('Video_After_Lane_detection.mp4', fourcc, 30, size)
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
video.release()
