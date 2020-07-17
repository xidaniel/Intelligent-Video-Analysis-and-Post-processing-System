# A Hybrid Method for Extracting and Reoframing Objects in Lecture Video       

# Abstract
In this project, We will use Mask-RCNN algorithm and OpenCV framework to analyze lecture video objects.
We will use Mask-RCNN to detect person and screen, then combine them to a new layout without pixel loss.

# The Architecture of Project
<img src="https://github.com/xidaniel/Lecture-Video-Objects-Reframing/blob/master/image/structure%20of%20project.png" width=800 align=center />

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

<img src="https://github.com/xidaniel/Lecture-Video-Objects-Reframing/blob/master/image/example.png" width = "400"  alt="Original" align=center />

# Future Work

- Inpainting algorithm optimization
- Contour algorithm optimization
- Multiple people and screen situation
- Tracker optimization
- etc.
