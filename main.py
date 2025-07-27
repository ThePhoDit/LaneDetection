import cv2
import numpy as np
from typing import Optional, Tuple, List

def canny(image: np.ndarray) -> np.ndarray:
    """
    Applies Canny edge detection to the input image.

    Parameters:
        image (np.ndarray): Input image in RGB format.

    Returns:
        np.ndarray: Image with edges detected.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    smooth_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    return cv2.Canny(smooth_image, 50, 150)

def average_slope_intercept(lines: Optional[np.ndarray]) -> List[np.ndarray]:
    """
    Calculates the average slope and intercept of the detected lines and returns extrapolated line coordinates.

    Parameters:
        lines (np.ndarray | None): Detected lines from Hough Transform.

    Returns:
        List[np.ndarray]: List of extrapolated lines as (x1, y1, x2, y2).
    """
    if lines is None:
        return []

    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue  # skip vertical lines to avoid division by zero
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

    height = 720  # adjust based on actual image size if needed
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
    """
    Applies a triangular mask to keep only the region of interest in the image.

    Parameters:
        image (np.ndarray): Input image.

    Returns:
        np.ndarray: Masked image.
    """
    height = image.shape[0]
    polygon = np.array([[(200, height), (1100, height), (550, 250)]], dtype=np.int32)
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(image, mask)

def display_lines(image: np.ndarray, lines: List[np.ndarray]) -> np.ndarray:
    """
    Draws extrapolated lines on the input image.

    Parameters:
        image (np.ndarray): Original image.
        lines (List[np.ndarray]): List of lines to draw.

    Returns:
        np.ndarray: Image with lines drawn.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def main() -> None:
    image_path = "media/test_image.jpg"
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canny_edges = canny(rgb_image)
    roi_image = region_of_interest(canny_edges)

    lines = cv2.HoughLinesP(
        roi_image, rho=2, theta=np.pi / 180, threshold=100,
        minLineLength=100, maxLineGap=50
    )

    averaged_lines = average_slope_intercept(lines)
    line_image = display_lines(image, averaged_lines)
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)

    cv2.imshow("Detected Lanes", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture("media/test2.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        canny_edges = canny(rgb_frame)
        roi_frame = region_of_interest(canny_edges)

        lines = cv2.HoughLinesP(
            roi_frame, rho=2, theta=np.pi / 180, threshold=100,
            minLineLength=100, maxLineGap=50
        )

        averaged_lines = average_slope_intercept(lines)
        line_image = display_lines(frame, averaged_lines)
        combined_frame = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        cv2.imshow("Detected Lanes", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
