# A Hybrid Method for Extracting and Reorganization Objects in Lecture Video       

# Abstract
In this project, We will use Mask-RCNN algorithm and OpenCV framework to analyze lecture video objects.
We will use Mask-RCNN to detect person and screen, then combine them to a new layout without pixel loss.

# What techniques I used
- Transfer Learning
- Train model on GPU of CGP
- Contour detection
- Find the corner vertices of quadrilateral
- Perspective transformation
- Tracker
- Separate image and audio from video
- Image fusion
- Color space select
- etc.

# Currently result for this project
Let's see the original video and outputs.

<img src="https://github.com/xidaniel/Video-analysis-and-algorithm-optimization/blob/master/image/original.png" width = "400"  alt="Original" align=center />

<img src="https://github.com/xidaniel/Video-analysis-and-algorithm-optimization/blob/master/image/no%20person.png" width = "400"  alt="no person" align=center />

<img src="https://github.com/xidaniel/Video-analysis-and-algorithm-optimization/blob/master/image/person%20in%20right.png" width = "400"  alt="Person in right" align=center />

<img src="https://github.com/xidaniel/Video-analysis-and-algorithm-optimization/blob/master/image/person%20in%20left.png" width = "400"  alt="Person in left" align=center />

# Future Work

- Inpainting algorithm optimization
- Contour algorithm optimization
- Multiple people and screen situation
- Tracker optimization
- etc.
