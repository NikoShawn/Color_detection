#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ColorDetection:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 红色范围定义
        self.lower_red = np.array([0, 100, 100])     # 红色范围低阈值
        self.upper_red = np.array([10, 255, 255])    # 红色范围高阈值
        
        # 字体设置
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 图片保存计数器
        self.num = 0
        
        # 订阅颜色图像话题
        self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        rospy.loginfo("红色物体检测节点已启动，订阅话题: /camera/color/image_raw")

    def image_callback(self, msg):
        """处理接收到的图像消息"""
        try:
            # 将ROS Image消息转换为OpenCV图像
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge错误: {e}")
            return
        
        # 红色检测处理
        self.process_frame(frame)

    def process_frame(self, frame):
        """处理单帧图像进行红色检测"""
        # 转换为HSV颜色空间
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 根据红色范围创建掩码
        mask_red = cv2.inRange(hsv_img, self.lower_red, self.upper_red)
        
        # 中值滤波去噪
        mask_red = cv2.medianBlur(mask_red, 7)
        
        # 查找轮廓
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 找到面积最大的红色物体
        max_area = 0
        largest_contour = None
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if area > 500 and area > max_area:  # 过滤掉太小的区域并找到最大面积
                max_area = area
                largest_contour = cnt
        
        # 只绘制面积最大的红色物体检测框
        if largest_contour is not None:
            (x, y, w, h) = cv2.boundingRect(largest_contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, f"Red (Area: {int(max_area)})", (x, y - 5), self.font, 0.7, (0, 0, 255), 2)
        
        # 更新图片计数器
        self.num += 1
        
        # 显示结果
        cv2.imshow("Red Object Detection", frame)
        
        # 保存图片（可选，创建imgs文件夹）
        try:
            cv2.imwrite(f"imgs/{self.num}.jpg", frame)
        except:
            pass  # 如果imgs文件夹不存在，跳过保存
        
        # 检查是否按下ESC键退出
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC键
            rospy.signal_shutdown("用户按下ESC键退出")

    def cleanup(self):
        """清理资源"""
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('red_detection_node', anonymous=True)
    
    try:
        # 创建红色检测实例
        detector = ColorDetection()
        
        # 保持节点运行
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("红色检测节点被中断")
    except KeyboardInterrupt:
        rospy.loginfo("收到键盘中断信号")
    finally:
        # 清理资源
        cv2.destroyAllWindows()
        rospy.loginfo("红色检测节点已关闭")