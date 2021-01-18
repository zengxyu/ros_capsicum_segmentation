#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from utils.util_boolean_array import serialize_boolean_array,deserialize_boolean_array
import numpy as np
import sys
# sys.path.append("/home/zeng/catkin_ws/install/lib/python3/dist-packages/ros_capsicum_segmentation")
# camera
from lib.msg import Mask
from threading import Thread,Lock

class ImagePublisher:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("image", Image, queue_size=10)
        self.mask_sub = rospy.Subscriber("mask", Mask, callback=self.mask_sub_callback)


    def image_publish(self, path):
        image = cv2.imread(path)
        print("image shape : ", image.shape)
        img_msg = None
        try:
            img_msg = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        except CvBridgeError as e:
            print(e)
        self.image_pub.publish(img_msg)
        print("image publisher sent an image")

    def mask_sub_callback(self, mask):
        print("mask subscriber received an image")
        height, width, mask_result = mask.height, mask.width, mask.data
        mask_deserialized = deserialize_boolean_array(mask_result,[300,400,3])
        mask = np.reshape(mask_deserialized,[300,400,3]).astype(np.uint8)
        mask[mask==1] = 255
        cv2.imshow("image",mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# def wait():
#     mask_res = rospy.wait_for_message('k', Mask, timeout=None)
#     image_pub.mask_sub_callback(mask_res)

if __name__ == '__main__':
    rospy.init_node('image_publisher', anonymous=True)
    image_pub = ImagePublisher()
    rate = rospy.Rate(1)

    # py.spin()
    thread = Thread(target=rospy.spin)
    thread.start()
    print(".....")

    path = "../assets/images/frame0000.jpg"
    while not rospy.is_shutdown():
        image_pub.image_publish(path)
        rate.sleep()


