#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class VideoSaver:
    def __init__(self):
        # 初始化CvBridge
        self.bridge = CvBridge()
        
        # VideoWriter相关设置
        self.video_writer = None
        self.output_video_path = "camera_output.mp4"  # 输出视频文件名
        self.video_fps = 30.0  # 视频帧率
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编解码器
        
        # 订阅相机图像话题
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        rospy.loginfo(f"开始订阅图像话题: /camera/color/image_raw")
        rospy.loginfo(f"视频将保存为: {self.output_video_path}")

    def image_callback(self, msg):
        """处理接收到的图像消息"""
        try:
            # 将ROS Image消息转换为OpenCV图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge错误: {e}")
            return
        
        # 初始化VideoWriter（第一次接收图像时）
        if self.video_writer is None:
            height, width, channels = cv_image.shape
            self.video_writer = cv2.VideoWriter(
                self.output_video_path, 
                self.fourcc, 
                self.video_fps, 
                (width, height)
            )
            
            if self.video_writer.isOpened():
                rospy.loginfo(f"VideoWriter初始化成功，视频尺寸: {width}x{height}")
            else:
                rospy.logerr("VideoWriter初始化失败")
                return
        
        # 将当前帧写入视频文件
        if self.video_writer is not None and self.video_writer.isOpened():
            self.video_writer.write(cv_image)
        
        # 可选：显示当前帧（用于调试）
        # cv2.imshow("Camera Feed", cv_image)
        # cv2.waitKey(1)

    def cleanup(self):
        """清理资源"""
        if self.video_writer is not None:
            rospy.loginfo("正在保存并关闭视频文件...")
            self.video_writer.release()
        cv2.destroyAllWindows()
        rospy.loginfo("视频保存完成")

if __name__ == '__main__':
    # 初始化ROS节点
    rospy.init_node('video_saver_node', anonymous=True)
    
    video_saver = None
    try:
        # 创建VideoSaver实例
        video_saver = VideoSaver()
        
        # 保持节点运行
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("视频保存节点被中断")
    except KeyboardInterrupt:
        rospy.loginfo("收到键盘中断信号")
    finally:
        # 清理资源
        if video_saver is not None:
            video_saver.cleanup()
        else:
            cv2.destroyAllWindows()
        rospy.loginfo("视频保存节点已关闭")