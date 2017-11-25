# Faster_RCNN_as_ROS_Server
### Faster_RCNN_as_ROS_Server

This is a ROS Server served as an Object Detection using Faster_RCNN Method. Given an image and then output bbox information.

This idea comes form [FasterRCNN](https://arxiv.org/pdf/1506.01497.pdf)

This code is modified from [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn), but I modified it as a ROS server to do specific task for object Recognition in [PROGRESSLAB](http://progress.eecs.umich.edu/)

### DEMO Instruction

To run this demo, you have to install caffe_model as mentioned in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) in scripts folder and include a pretrained .caffemodel file. For model to download, please contact me ywchow@umich.edu

#### Detail Usage
1. Please recursive clone my repo, make sure you have included caffe-fast-rcnn folder in scripts folder 
2. In scripts/caffe-fast-rcnn folder, make sure you have changed the branch to faster-rcnn branch, not master branch
3. In scripts/caffe-fast-rcnn folder, make a copy of Makefile.config.example as Make.config and uncomment the line: WITH_PYTHON_LAYER := 1
4. In scripts/caffe-fast-rcnn folder, add opencv_imgcodecs in file Makefile. It is around the line 174 in LIBRARIES variable. Then run ``make`` and ``make python``. And pip install *scikit-image*, *easydict*, *protobuf*, *cython*, apt-get install python-opencv
5. In scripts/data/demo folder, include test images and make sure in client.py in scripts folder has specified the path of these test images.
6. In scripts/data/faster_rcnn_models, include pretrained model and make sure in demo.py in scripts folder has specified the path of pretrained model.
7. Run demo.py first, then run client.py

The demo video is shown on following youtube link [demo](https://www.youtube.com/watch?v=3dvnhPWKLrA)

### Use demo.py
Use `--net NET_NAME` to indicate which trained model to use, as being specified in
```
NETS = {key: (name of folder, model name, pre-trained or not, tuples of class)}
```

