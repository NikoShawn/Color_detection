#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from geometry_msgs.msg import PointStamped,TransformStamped,Point
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

class ColorDetection:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 红色范围定义
        # self.lower_red = np.array([156, 100, 100])
        # self.upper_red = np.array([180, 255, 255])
        
        self.lower_red = np.array([0, 150, 150])     # 红色范围低阈值
        self.upper_red = np.array([10, 180, 180])    # 红色范围高阈值

        # 字体设置
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 图片保存计数器
        self.num = 0
        
        # 订阅颜色图像和深度图像话题
        self.color_sub = message_filters.Subscriber('/cam_3/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/cam_3/aligned_depth_to_color/image_raw', Image) 

        # 优化1: 减小队列大小和时间差，增加处理频率
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 
            queue_size=5,  # 减小队列大小，从10改为5
            slop=0.05      # 减小时间差容忍度，从0.1改为0.05
        )
        self.ts.registerCallback(self.synchronized_callback)

        # 订阅相机内参
        self.camera_info_sub = rospy.Subscriber('/cam_3/color/camera_info', CameraInfo, self.camera_info_callback)
        
        # 添加发布器
        self.camera_coords_pub = rospy.Publisher('/target_obs', PointStamped, queue_size=5)  # 减小队列
        self.world_coords_pub = rospy.Publisher('/target_pos', PointStamped, queue_size=5)   # 减小队列
        
        # 相机内参变量
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_received = False
        
        # 添加相机坐标存储变量
        self.camera_coords = None

        # 订阅机械犬里程计
        self.odom_sub = rospy.Subscriber('/Odometry', Odometry, self.odom_callback)

        # 当前位姿存储
        self.current_pose = None
        self.orientation = None
        self.position = None
        
        # 优化2: 添加处理频率控制
        self.last_process_time = rospy.Time.now()
        self.process_interval = rospy.Duration(0.02)  # 约30Hz，可调整
        
        # 优化3: 添加图像降采样标志
        self.downsample_factor = 4  # 图像降采样因子，2表示宽高各缩小一半

        # 添加帧跳过计数器
        self.frame_skip_counter = 0
        self.process_every_n_frames = 3  # 每3帧处理一次，约10Hz

    def synchronized_callback(self, color_msg, depth_msg):
        """处理接收到的同步颜色和深度图像消息"""
        # 方法2: 每N帧处理一次
        self.frame_skip_counter += 1
        if self.frame_skip_counter < self.process_every_n_frames:
            return
        self.frame_skip_counter = 0
        
        try:
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1") 
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge错误: {e}")
            return
        
        # 红色检测和深度处理
        self.process_frame_and_depth(color_frame, depth_frame)

    def camera_info_callback(self, msg):
        """处理相机内参回调"""
        if not self.camera_info_received:
            self.fx = msg.K[0]
            self.fy = msg.K[4]
            self.cx = msg.K[2]
            self.cy = msg.K[5]
            self.camera_info_received = True
            rospy.loginfo(f"接收到相机内参: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
    
    def odom_callback(self, msg):
        """里程计回调更新当前位姿"""
        self.current_pose = msg.pose.pose
        self.position = [
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_pose.position.z
        ]
        self.orientation = [
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        ]

    def transform_to_world(self, camera_coords):
        """坐标转换函数"""
        if self.current_pose is None:
            rospy.logwarn("No odometry data received yet!")
            return None

        try:
            robot_x = self.position[0]
            robot_y = self.position[1]
            robot_z = self.position[2]

            q = self.orientation
            roll, pitch, yaw = euler_from_quaternion([q[0], q[1], q[2], q[3]])

            x_local = camera_coords[0]
            z_local = camera_coords[2]

            world_x = robot_x + z_local * np.cos(yaw) + x_local * np.sin(yaw)
            world_y = robot_y + z_local * np.sin(yaw) - x_local * np.cos(yaw)
            world_z = robot_z

            result = PointStamped()
            result.header = Header()
            result.header.stamp = rospy.Time.now()
            result.header.frame_id = "odom"
            result.point = Point(world_x, world_y, world_z)
            return result

        except Exception as e:
            rospy.logerr(f"Coordinate transform failed: {str(e)}")
            return None

    def pixel_to_3d(self, u, v, depth):
        """将像素坐标和深度值转换为三维坐标"""
        if not self.camera_info_received:
            rospy.logwarn("相机内参尚未接收，无法计算三维坐标")
            return None
        
        if depth <= 0:
            return None
        
        depth_m = depth / 1000.0
        
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        
        return x, y, z

    def process_frame_and_depth(self, color_frame, depth_frame):
        """处理单帧图像进行红色检测并获取深度信息"""
        # 优化5: 图像降采样以提高处理速度
        if self.downsample_factor > 1:
            height, width = color_frame.shape[:2]
            new_width = width // self.downsample_factor
            new_height = height // self.downsample_factor
            
            # 降采样图像
            small_color = cv2.resize(color_frame, (new_width, new_height))
            small_depth = cv2.resize(depth_frame, (new_width, new_height))
            
            # 相应调整内参
            scale_x = new_width / width
            scale_y = new_height / height
        else:
            small_color = color_frame
            small_depth = depth_frame
            scale_x = scale_y = 1.0
        
        # 转换为HSV颜色空间
        hsv_img = cv2.cvtColor(small_color, cv2.COLOR_BGR2HSV)
        
        # 根据红色范围创建掩码
        mask_red = cv2.inRange(hsv_img, self.lower_red, self.upper_red)
        
        # 优化6: 减少滤波核大小
        mask_red = cv2.medianBlur(mask_red, 5)  # 从7改为5
        
        # 查找轮廓
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 使用SIMPLE近似
        
        # 找到面积最大的红色物体
        max_area = 0
        largest_contour = None
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            # 优化7: 根据降采样调整最小面积阈值
            min_area = 1000 // (self.downsample_factor ** 2)
            if area > min_area and area > max_area:
                max_area = area
                largest_contour = cnt
        
        # 只绘制面积最大的红色物体检测框并获取深度
        if largest_contour is not None:
            (x, y, w, h) = cv2.boundingRect(largest_contour)
            
            # 如果使用了降采样，需要将坐标映射回原图
            if self.downsample_factor > 1:
                x = int(x / scale_x)
                y = int(y / scale_y)
                w = int(w / scale_x)
                h = int(h / scale_y)
            
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(color_frame, (center_x, center_y), 5, (255, 0, 0), -1)

            depth_value_mm = 0
            depth_text = "Depth: N/A"
            
            if 0 <= center_y < depth_frame.shape[0] and 0 <= center_x < depth_frame.shape[1]:
                depth_value_mm = depth_frame[center_y, center_x]
                
                if depth_value_mm > 0:
                    depth_value_m = depth_value_mm / 1000.0
                    depth_text = f"Depth: {depth_value_m:.2f}m"

                    camera_coords_tuple = self.pixel_to_3d(center_x, center_y, depth_value_mm)
                    if camera_coords_tuple:
                        x_cam, y_cam, z_cam = camera_coords_tuple
                        
                        camera_point_stamped = PointStamped()
                        camera_point_stamped.header.stamp = rospy.Time.now()
                        camera_point_stamped.header.frame_id = "camera_color_optical_frame" 
                        camera_point_stamped.point = Point(x_cam, y_cam, z_cam)
                        self.camera_coords_pub.publish(camera_point_stamped)
                        
                        # 优化8: 减少日志输出频率
                        if self.num % 10 == 0:  # 每10帧输出一次日志
                            rospy.loginfo(f"目标在相机坐标系下: x={x_cam:.2f}, y={y_cam:.2f}, z={z_cam:.2f}")

                        self.camera_coords = [x_cam, y_cam, z_cam] 

                        world_point_stamped = self.transform_to_world(self.camera_coords)
                        if world_point_stamped:
                            self.world_coords_pub.publish(world_point_stamped)
                            if self.num % 10 == 0:
                                rospy.loginfo(f"目标在世界坐标系下: x={world_point_stamped.point.x:.2f}, y={world_point_stamped.point.y:.2f}, z={world_point_stamped.point.z:.2f}")
                else:
                    depth_text = "Depth: 0mm (Invalid)"
            else:
                depth_text = "Depth: OOB"

            cv2.putText(color_frame, f"Red (Area: {int(max_area * (self.downsample_factor ** 2))})", (x, y - 25), self.font, 0.7, (0, 0, 255), 2)
            cv2.putText(color_frame, depth_text, (x, y - 5), self.font, 0.7, (0, 255, 0), 2)
        
        self.num += 1
        
        # 优化9: 降低显示帧率
        # if self.num % 2 == 0:  # 每2帧显示一次
        #     cv2.imshow("Red Object Detection", color_frame)
        
        # 优化10: 减少waitKey调用
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            rospy.signal_shutdown("用户按下ESC键退出")

    def cleanup(self):
        """清理资源"""
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('red_detection_node', anonymous=True)
    
    try:
        detector = ColorDetection()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("红色检测节点被中断")
    except KeyboardInterrupt:
        rospy.loginfo("收到键盘中断信号")
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("红色检测节点已关闭")