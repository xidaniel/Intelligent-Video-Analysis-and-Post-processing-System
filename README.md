# Lecture Video Transformation Through an Intellgent Video Analysis and Post-Processing System    
## About This Project
Lecture videos are good sources to learn something new. However, the videos without post-processing are usually posted on the internet directly due to the technology and resources limitations. We usually get highly-quality by consuming more time and resources. We investigate the problem of automatically extracting key information from a lecture video in different scenarios. The goal of this thesis is to extract main instructor and screen from a original live lecture video and then to combine them to generate a new layout-video with less file size than original.
## Demo
<img src="image/demo.gif" width = "700"  alt="Original" align=center />

## The Project Flow Diagram
<img src="image/archi.png" width=800 align=center />

## Real Time Architecture
<img src="image/real-time.png" width=800 align=center />

## Contributing
- Built an intelligent system to detect objects, filter objects, track objects, render new video, etc., utilizing Python, OpenCV.
- Collected 8,100+ images and Trained an object detection model to localize person, face, and screen on a Mask R-CNN framework.  
- Proposed an algorithm to identify the main speaker in a multi-people scenario based on face, body detection, and video sequence.
- Investigated an algorithm to get screen contour in pixel level, correct screen viewpoint, and enhance screen visual quality.
- Accelerated the video analysis and rendering speed by 35x via integrating the object detection model and a tracking algorithm.


## Techniques In Project
- Deep Learning (Computer Vision)
    - Transfer Learning
    - Object Detection
    - Instance Segmentation
    - ROI Behavior Analysis
    - Train model on multi GPUs
- Image Processing
    - Contour detection
    - Find the corner vertices of quadrilateral
    - Design corner filter algorithm for screen
    - Perspective transformation
    - Tracker
    - Image fusion
    - Color space select
- Video Utils
    - Split and combine frame and audio from video

## Future Work
- Inpainting algorithm optimization
- Contour algorithm optimization
- Mouth motion detection
- Tracker optimization
- etc.

## Contact
Email: xiwang3317@gmail.com
