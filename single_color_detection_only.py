#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header

class ColorDetection:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 红色范围定义
        self.lower_red = np.array([165, 100, 100])     # 红色范围低阈值
        self.upper_red = np.array([180, 255, 255])    # 红色范围高阈值
        
        # 字体设置
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 图片保存计数器
        self.num = 0
        
        # 订阅彩色图像话题
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        # 添加发布器 - 红色目标检测结果
        self.detection_pub = rospy.Publisher('/red_person_detection', PointStamped, queue_size=10)
        
        # 发布处理后的彩色图像
        self.processed_image_pub = rospy.Publisher('/processed_color_image', Image, queue_size=10)

    def image_callback(self, msg):
        """处理接收到的图像消息"""
        try:
            # 将ROS Image消息转换为OpenCV图像
            color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge错误: {e}")
            return
        
        # 红色检测处理
        self.process_frame(color_frame)

    def process_frame(self, color_frame):
        """处理单帧图像进行红色检测"""
        # 转换为HSV颜色空间
        hsv_img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        
        # 根据红色范围创建掩码
        mask_red = cv2.inRange(hsv_img, self.lower_red, self.upper_red)
        
        # 中值滤波去噪
        mask_red = cv2.medianBlur(mask_red, 7)
        
        # 查找红色轮廓
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 处理红色检测
        max_red_area = 0
        largest_red_contour = None
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if area > 100 and area > max_red_area:  # 过滤掉太小的区域并找到最大面积
                max_red_area = area
                largest_red_contour = cnt
        
        # 只绘制面积最大的红色物体检测框
        if largest_red_contour is not None:
            (x, y, w, h) = cv2.boundingRect(largest_red_contour)
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # 计算检测框中心点
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(color_frame, (center_x, center_y), 5, (255, 0, 0), -1) # 绘制中心点

            # 发布检测结果（像素坐标）
            detection_point = PointStamped()
            detection_point.header = Header()
            detection_point.header.stamp = rospy.Time.now()
            detection_point.header.frame_id = "camera_color_optical_frame"
            detection_point.point = Point(center_x, center_y, max_red_area)  # 使用z坐标存储面积信息
            self.detection_pub.publish(detection_point)

            cv2.putText(color_frame, f"Red Person (Area: {int(max_red_area)})", (x, y - 10), self.font, 0.7, (0, 0, 255), 2)
            cv2.putText(color_frame, f"Center: ({center_x}, {center_y})", (x, y + h + 20), self.font, 0.5, (0, 255, 0), 2)
            
            rospy.loginfo(f"检测到红色人员，中心坐标: ({center_x}, {center_y}), 面积: {int(max_red_area)}")
            
        # 发布处理后的彩色图像作为ROS话题
        try:
            processed_image_msg = self.bridge.cv2_to_imgmsg(color_frame, "bgr8")
            processed_image_msg.header.stamp = rospy.Time.now()
            processed_image_msg.header.frame_id = "camera_color_optical_frame"
            self.processed_image_pub.publish(processed_image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"发布处理后图像时出错: {e}")
        
        # 更新图片计数器
        self.num += 1
        
        # 检查是否按下ESC键退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            rospy.signal_shutdown("用户按下ESC键退出")

    def cleanup(self):
        """清理资源"""
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('red_person_detection_node', anonymous=True)
    
    try:
        # 创建红色检测实例
        detector = ColorDetection()
        
        # 保持节点运行
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("红色人员检测节点被中断")
    except KeyboardInterrupt:
        rospy.loginfo("收到键盘中断信号")
    finally:
        # 清理资源
        cv2.destroyAllWindows()
        rospy.loginfo("红色人员检测节点已关闭")