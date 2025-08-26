#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header

class ColorDetection(Node):
    def __init__(self):
        super().__init__('red_person_detection_node')
        self.bridge = CvBridge()
        
        # 红色范围定义
        self.lower_red = np.array([0, 150, 150])
        self.upper_red = np.array([10, 180, 180])
        
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.num = 0

        # 订阅彩色图像话题
        self.color_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        
        # 发布红色目标检测结果
        self.detection_pub = self.create_publisher(
            PointStamped,
            '/red_person_detection',
            10
        )
        
        # 发布处理后的彩色图像
        self.processed_image_pub = self.create_publisher(
            Image,
            '/processed_color_image',
            10
        )

    def image_callback(self, msg):
        try:
            color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"CvBridge错误: {e}")
            return
        self.process_frame(color_frame)

    def process_frame(self, color_frame):
        hsv_img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv_img, self.lower_red, self.upper_red)
        mask_red = cv2.medianBlur(mask_red, 7)
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_red_area = 0
        largest_red_contour = None
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if area > 50 and area > max_red_area:
                max_red_area = area
                largest_red_contour = cnt
        if largest_red_contour is not None:
            (x, y, w, h) = cv2.boundingRect(largest_red_contour)
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(color_frame, (center_x, center_y), 5, (255, 0, 0), -1)
            detection_point = PointStamped()
            detection_point.header.stamp = self.get_clock().now().to_msg()
            detection_point.header.frame_id = "camera_color_optical_frame"
            detection_point.point = Point(
                x=float(center_x),
                y=float(center_y),
                z=float(max_red_area)
            )
            self.detection_pub.publish(detection_point)
            cv2.putText(color_frame, f"Red Person (Area: {int(max_red_area)})", (x, y - 10), self.font, 0.7, (0, 0, 255), 2)
            cv2.putText(color_frame, f"Center: ({center_x}, {center_y})", (x, y + h + 20), self.font, 0.5, (0, 255, 0), 2)
            self.get_logger().info(f"检测到红色人员，中心坐标: ({center_x}, {center_y}), 面积: {int(max_red_area)}")
        try:
            processed_image_msg = self.bridge.cv2_to_imgmsg(color_frame, "bgr8")
            processed_image_msg.header.stamp = self.get_clock().now().to_msg()
            processed_image_msg.header.frame_id = "camera_color_optical_frame"
            self.processed_image_pub.publish(processed_image_msg)
        except Exception as e:
            self.get_logger().error(f"发布处理后图像时出错: {e}")
        self.num += 1

    def cleanup(self):
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    detector = ColorDetection()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info("收到键盘中断信号")
    finally:
        detector.cleanup()
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()