# Lane Detection

This repository contains code for lane detection using computer vision techniques. The project is designed to detect lanes in images and videos, which can be useful for autonomous driving systems.

It uses OpenCV for image processing and employs techniques such as edge detection, Hough Transform, and region of interest masking to identify lane lines in images.

## Requirements
- Python 3.x
- OpenCV
- NumPy

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/ThePhoDit/LaneDetection.git
   ```
   
2. Navigate to the project directory:
   ```bash
   cd LaneDetection
   ```
   
3. Install the required packages:
   ```bash
    pip install -r requirements.txt
    ```
4. Run the lane detection script:
   ```bash
   python main.py --mode <image/video> --input <path_to_image_or_video>
   ```
   
> [!TIP]
> Keep in mind that a mask is applied to the image to focus on the region of interest, which is typically the lower half of the image where lanes are usually located. You can adjust the mask coordinates in the code if needed.

This code was created following the tutorial from [Programming Knowledge](https://www.youtube.com/watch?v=eLTLtUVuuy4&list=PLZ9yCGFMeMboFcNF55cEuAKBkME5Wcls8) and is intended for educational purposes. Feel free to modify and adapt it for your own projects.