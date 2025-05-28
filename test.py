#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class RedPixelDetector:
    def __init__(self):
        rospy.init_node('red_pixel_detector', anonymous=True)
        
        self.bridge = CvBridge()
        
        # 设置用于识别红色的HSV阈值
        self.lower_red1 = np.array([0, 100, 100]) # 0
        self.upper_red1 = np.array([5, 255, 255]) # 10
        self.lower_red2 = np.array([180, 100, 100]) # 170
        self.upper_red2 = np.array([180, 255, 255]) # 180
        
        # 订阅彩色图像
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.callback)
        
        rospy.loginfo("Red pixel detector node initialized")
    
    def callback(self, color_msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            color_img = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            
            # 转换到HSV色彩空间以便更容易识别红色
            hsv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
            
            # 创建红色掩码 (红色在HSV中分布在两个范围)
            mask1 = cv2.inRange(hsv_img, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv_img, self.lower_red2, self.upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            
            # 获取红色像素的坐标
            red_pixels = np.where(red_mask > 0)
            
            # 如果有红色像素
            if len(red_pixels[0]) > 0:
                pixel_count = len(red_pixels[0])
                rospy.loginfo(f"Detected {pixel_count} red pixels")
                
                # 可视化结果
                self.visualize_results(color_img, red_mask, pixel_count)
            else:
                rospy.loginfo("No red pixels detected")
                # 显示原图
                cv2.imshow("Red Pixels Detection", color_img)
                cv2.waitKey(1)
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {str(e)}")
            import traceback
            rospy.logerr(traceback.format_exc())
    
    def visualize_results(self, color_img, red_mask, pixel_count):
        # 在原图上标记红色区域
        result = color_img.copy()
        result[red_mask > 0] = [0, 0, 255]  # 将红色区域标记为纯红色
        
        # 添加文本信息
        cv2.putText(result, f"Red Pixels: {pixel_count}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow("Red Pixels Detection", result)
        cv2.waitKey(1)

if __name__ == '__main__':
    try:
        detector = RedPixelDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()