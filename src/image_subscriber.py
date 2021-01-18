import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from src.image_predictor import get_args, predict_image,compare_pred_mask_and_ground_truth
from lib.msg import Mask
import numpy as np
from utils.util_boolean_array import serialize_boolean_array

class ImageSubscriber:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, callback=self.image_sub_callback)

        # self.image_sub = rospy.Subscriber("image", Image, callback=self.image_sub_callback)
        self.mask_pub = rospy.Publisher("mask", Mask, queue_size=10)

    def mask_publish(self, mask_result):
        mask_result = np.array(mask_result).astype(np.uint8)
        shape = mask_result.shape
        print("mask shape : ", shape)

        mask = Mask()
        mask.height, mask.width = shape[0], shape[1]
        mask.data = serialize_boolean_array(mask_result)


        self.mask_pub.publish(mask)
        print("mask publisher sent an image")


    def image_sub_callback(self, image):
        print("image_subscriber received an image")
        cv_image = None
        # image = image.data
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)
        pred_mask = self.predict(cv_image)
        self.mask_publish(np.array(pred_mask))
        # cv2.imshow("subscribed image", cv_image)
        # cv2.waitKey(3)
        # cv2.destroyAllWindows()

    def predict(self, cv_image):
        args = get_args()
        pred_mask, gt_image = predict_image(args, cv_image, cv_image)
        # pred_mask = np.array(pred_mask)
        compare_pred_mask_and_ground_truth(args, pred_mask, gt_image, True)
        return pred_mask

if __name__ == '__main__':
    rospy.init_node("image_subscriber")
    sub = ImageSubscriber()
    rospy.spin()
