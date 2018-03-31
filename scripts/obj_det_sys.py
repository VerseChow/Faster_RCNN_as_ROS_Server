import _init_paths

from faster_rcnn_object_detector.srv import *    
from faster_rcnn_object_detector.msg import *
import rospy


from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import glob


class obj_detection_system:

    #CLASSES = cfg.CLASSES
    #print CLASSES

    #CLASSES = ('__background__', 'person', 'bicycle', 'car','motorcycle','airplane','bus','train', 'truck', 'boat','traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench','bird','cat','dog','horse','sheep','cow', 'elephant','bear','zebra','giraffe','hat','umbrella', 'handbag','tie','suitcase', 'frisbee','skis','snowboard','sports ball','kite', 'baseball bat','baseball glove','skateboard','surfboard','tennis racket', 'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich', 'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant', 'bed','dining table','window','tv','laptop','mouse','remote','keyboard','cell phone','microwave', 'oven', 'sink','refrigerator','blender','book','clock','vase','scissors','teddy bear','hair drier','tooth brush')
    #print CLASSES
    #sys.exit(0)

    def __init__(self, net, gpu_flag = True, gpu_device = 0, CONF_THRESH = 0.8, NMS_THRESH = 0.005):
        self.net = net
        self.CONF_THRESH = CONF_THRESH
        self.NMS_THRESH = NMS_THRESH
        self.gpu_flag = gpu_flag
        self.gpu_device = gpu_device
        self.CLASSES = cfg.CLASSES

    def get_bbox(self, im, class_name, dets, thresh=0.0):
        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return None
        bbox = np.empty((0,4), float)
        score = np.empty((0,1), float)
        for i in inds:
            bbox = np.append(bbox, np.array([dets[i, :4]]), axis = 0)
            score = np.append(score, dets[i, -1])
        
        #bbox = [xmin, ymin, xmax, ymax]
        inds = inds.tolist()
        return {'bbox': bbox, 'score': score, 'inds': inds}

    def vis_detections(self, im, class_name, dets, ax,thresh=0.0):

        inds = np.where(dets[:, -1] >= thresh)[0]
        if len(inds) == 0:
            return

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='red', linewidth=3.5)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(class_name, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

        plt.axis('off')
        plt.tight_layout()
        plt.draw()

    def demo(self, image_name):
        '''Demo for Object Detection in images of a folder'''
        """Detect object classes in an image using pre-computed object proposals."""
        # Load the demo image
        im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
        im = cv2.imread(im_file)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()

        scores, boxes = im_detect(self.net, im)

        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        # Visualize detections for each class


        fig, ax = plt.subplots(figsize=(12, 12))
        for cls_ind, cls in enumerate(self.CLASSES[1:]):

            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, self.NMS_THRESH)
            dets = dets[keep, :]
            bbox = self.get_bbox(im, cls, dets, thresh=self.CONF_THRESH)
            print 'Detected Object in '+cls
            print bbox
            self.vis_detections(im, cls, dets, ax, thresh=self.CONF_THRESH)
            
        im = im[:, :, (2, 1, 0)]
            
        ax.set_title(('class detections with '
                         'p(class | box) >= {:.1f}').format(self.CONF_THRESH),
                          fontsize=14)
        ax.imshow(im, aspect='equal')

        plt.savefig('./results/'+os.path.basename(im_file))

        return cv2.imread('./results/'+os.path.basename(im_file))
            

    def image_process(self, im):
        # Detect all object classes and regress object bounds
        resp = ImageToObjectResponse()

        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.net, im)

        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        for cls_ind, cls in enumerate(self.CLASSES[1:]):

            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            #print dets
            keep = nms(dets, self.NMS_THRESH)
            dets = dets[keep, :]
            detection_result = self.get_bbox(im, cls, dets, thresh=self.CONF_THRESH)
            print 'Detected Object in '+cls
            print detection_result
            
            if detection_result is not None:
                bbox = detection_result['bbox'] 
                score = detection_result['score']
                
                Object = ObjectInfo()
                Object.label = cls
                Object.all_label = self.CLASSES[1:]
                Object.bbox_xmin = bbox[:,0]
                Object.bbox_ymin = bbox[:,1]
                Object.bbox_xmax = bbox[:,2]
                Object.bbox_ymax = bbox[:,3]
                Object.score = score
                resp.objects.append(Object)

            print '***************************************'
            
        return resp
    
    def image_process_bbox_with_nms(self, im):
        # Detect all object classes and regress object bounds
        resp = ImageToBBoxResponse()

        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.net, im)

        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        for cls_ind, cls in enumerate(self.CLASSES[1:]):

            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            #print dets
            keep = nms(dets, self.NMS_THRESH)
            dets = dets[keep, :]
            detection_result = self.get_bbox(im, cls, dets, thresh=self.CONF_THRESH)
            print 'Detected Object in '+cls
            print detection_result
            
            if detection_result is not None:
                bbox = detection_result['bbox'] 
                score = detection_result['score']
                
                test_index = [keep[i] for i in detection_result['inds']]
                all_score = scores[test_index, 1:]
                
                i = 0
                for box in bbox:
                    BBox = BBoxInfo()
                    BBox.bbox_xmin = int(box[0])
                    BBox.bbox_ymin = int(box[1])
                    BBox.bbox_xmax = int(box[2])
                    BBox.bbox_ymax = int(box[3])
                    BBox.label = self.CLASSES[1:]
                    BBox.score = all_score[i,:].tolist()
                    BBox.max_label = cls
                    BBox.max_score = score[i]
                    
                    print BBox.label
                    print BBox.score
                    print BBox.max_score
                    
                    i=i+1
                    resp.bbox.append(BBox)

            print '***************************************'
            
        return resp
    
    def image_process_bbox_without_nms(self, im):
        # Detect all object classes and regress object bounds
        resp = ImageToBBoxResponse()

        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(self.net, im)
        
        scores = scores[:,1:]
        max_scores = np.max(scores, axis=1)

        timer.toc()
        print ('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0])

        boxes = boxes[:, 0:4]
        dets = np.hstack((boxes, max_scores[:, np.newaxis])).astype(np.float32)
        detection_result = self.get_bbox(im, '', dets, thresh=self.CONF_THRESH)
        print 'Detected BBox ' + str(len(detection_result['inds']))
        print detection_result
        
        if detection_result is not None:
            bbox = detection_result['bbox'] 
            test_index = detection_result['inds']
            all_score = scores[test_index, :]
            
            i = 0
            for box in bbox:
                BBox = BBoxInfo()
                BBox.bbox_xmin = int(box[0])
                BBox.bbox_ymin = int(box[1])
                BBox.bbox_xmax = int(box[2])
                BBox.bbox_ymax = int(box[3])
                BBox.label = self.CLASSES[1:]
                BBox.score = all_score[i,:].tolist()
                
                max_score_index = all_score[i,:].argmax()
                BBox.max_label = BBox.label[max_score_index]
                BBox.max_score = BBox.score[max_score_index]
                
                print BBox.label
                print BBox.score
                print BBox.max_score
                
                i=i+1
                resp.bbox.append(BBox)

            print '***************************************'
            
        return resp

    def handle_image_objects(self, req):
        print "Received Image, start Detection"
        image = req.image
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image, "bgr8")
        
        if self.gpu_flag:
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_device)
        else:
            caffe.set_mode_cpu()    
        
        return self.image_process(image)

    def handle_image_bbox(self, req):
        print "Received Image, start Detection"
        image = req.image
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image, "bgr8")
        
        if self.gpu_flag:
            caffe.set_mode_gpu()
            caffe.set_device(self.gpu_device)
        else:
            caffe.set_mode_cpu()    
        
        return self.image_process_bbox_with_nms(image) # prefer this one!
        #return self.image_process_bbox_without_nms(image) # give somewhat multiple bad bbox for one object instance

    def bbox_detection_server(self):
        rospy.init_node('object_detection_server')
        s = rospy.Service('object_detection', ImageToObject, self.handle_image_objects)
        #s = rospy.Service('object_detection', ImageToBBox, self.handle_image_bbox)
        print "Ready for Object Detection"
        rospy.spin()
