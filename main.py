import cv2
import numpy as np
import argparse
import os
from typing import Optional, Tuple, List


def canny(image: np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    smooth_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return cv2.Canny(smooth_image, 50, 150)


def average_slope_intercept(lines: Optional[np.ndarray]) -> List[np.ndarray]:
    if lines is None:
        return []

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    def make_line(slope: float, intercept: float, y1: int, y2: int) -> np.ndarray:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def average(lines: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if not lines:
            return None
        slope = np.mean([s for s, _ in lines])
        intercept = np.mean([i for _, i in lines])
        return slope, intercept

    height = 720
    y1 = height
    y2 = int(height * 0.6)

    averaged_lines = []
    left_avg = average(left_lines)
    if left_avg:
        averaged_lines.append(make_line(*left_avg, y1, y2))

    right_avg = average(right_lines)
    if right_avg:
        averaged_lines.append(make_line(*right_avg, y1, y2))

    return averaged_lines


def region_of_interest(image: np.ndarray) -> np.ndarray:
    height = image.shape[0]
    polygon = np.array([[(200, height), (1100, height), (550, 250)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)


def display_lines(image: np.ndarray, lines: List[np.ndarray]) -> np.ndarray:
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return line_image


def process_frame(frame: np.ndarray) -> np.ndarray:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    edges = canny(rgb_frame)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 2, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=50)
    averaged = average_slope_intercept(lines)
    line_image = display_lines(frame, averaged)
    return cv2.addWeighted(frame, 1, line_image, 0.7, 1)


def run_video(input_path: str) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {input_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed = process_frame(frame)
        cv2.imshow("Detected Lanes - Video", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_image(input_path: str) -> None:
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {input_path}")

    processed = process_frame(image)
    cv2.imshow("Detected Lanes - Image", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Lane Detection using OpenCV")
    parser.add_argument('--mode', type=str, choices=['image', 'video'], required=True,
                        help='Mode to run the detector: "image" or "video"')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or video file')
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    if args.mode == 'image':
        run_image(args.input)
    elif args.mode == 'video':
        run_video(args.input)
    else:
        raise ValueError("Invalid mode. Use 'image' or 'video'.")


if __name__ == "__main__":
    main()
