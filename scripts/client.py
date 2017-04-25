from obj_detection.srv import *    
from obj_detection.msg import *
import rospy
import cv2
import glob
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

def image_grab_client(im):


	rospy.wait_for_service('object_detection')
	bridge = CvBridge()
	im_msg = bridge.cv2_to_imgmsg(im, "bgr8")

	try:
		print "sent image to server"
		obj_detect_handle = rospy.ServiceProxy('object_detection', ImageToBBox)
		resp = obj_detect_handle(im_msg)
		if not resp:
			print 'nothing detected'
			return
		else:
			for bbox in resp.objects:
				txt = bbox.label
				for index in range(len(bbox.bbox_xmin)):
					draw_lines_text(im, bbox, txt, index)
			return im

	except rospy.ServiceException, e:
		print "Service call failed: %s"%e

def draw_lines_text(im, bbox, txt, index):
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.rectangle(im,(bbox.bbox_xmin[index],bbox.bbox_ymin[index]),(bbox.bbox_xmax[index],bbox.bbox_ymax[index]),(0,255,0),8)
	cv2.putText(im, txt, (bbox.bbox_xmin[index],bbox.bbox_ymin[index]), font, 4,(255,255,255),8,cv2.LINE_AA)


if __name__=="__main__":

	im_list = sorted(glob.glob("./data/demo/*.jpg"))
	 
	for im in im_list:
		cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
		
		image = cv2.imread(im)
			
		image = image_grab_client(image)
		cv2.imshow('image',image)
		cv2.waitKey(10)


		