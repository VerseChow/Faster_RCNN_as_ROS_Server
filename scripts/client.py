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
	im = bridge.cv2_to_imgmsg(im, "bgr8")
	try:
		print "sent image to server"
		obj_detect_handle = rospy.ServiceProxy('object_detection', ImageToBBox)
		resp = obj_detect_handle(im)
		print resp

	except rospy.ServiceException, e:
		print "Service call failed: %s"%e


if __name__=="__main__":

	im_list = glob.glob("./data/demo/*.jpg")

	for im in im_list:
		image = cv2.imread(im)
		image_grab_client(image)