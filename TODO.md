### familiarizing with object detection models such as [yolo](https://github.com/Hyouteki/BTP/tree/main/yolo), [tflite](https://github.com/Hyouteki/BTP/tree/main/tflite)
- detecting objects in a video and using that in furthur anomaly detection such as collisions/accidents, running in hallway, etc; which requires computation of interaction between objects not detection.

 ### socket programming 
- as some level of computation is required at the edge and raspberry pi is not capable of extended computations. so, using raspberry pi for the bare minimum task that is transferring of frames/segment of video over a network to the server.

### dataset training
- as pretrained models are not suitable for this niche. we require to further train those models using some available dataset comprising of images and cctv videos using the concept of transfer learning. 
- also training a model is a expensive job we have to relay on softwares like google colab which provides GPU/TPU.

### testing

### further improvements
- not just limiting this idea to cctv cameras but extending it to mobile/webcams.
- multiple camera feed for improved accurcy.
