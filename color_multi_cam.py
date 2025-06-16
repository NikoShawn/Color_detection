#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from geometry_msgs.msg import PointStamped, TransformStamped, Point
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

class MultiCameraColorDetection:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 红色范围定义
        self.lower_red = np.array([156, 100, 100])
        self.upper_red = np.array([180, 255, 255])
        
        # 字体设置
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 图片保存计数器
        self.num = 0
        
        # 前摄像头订阅
        self.front_color_sub = message_filters.Subscriber('/cam_1/color/image_raw', Image)
        self.front_depth_sub = message_filters.Subscriber('/cam_1/aligned_depth_to_color/image_raw', Image)
        
        # 后摄像头订阅
        self.rear_color_sub = message_filters.Subscriber('/cam_2/color/image_raw', Image)
        self.rear_depth_sub = message_filters.Subscriber('/cam_2/aligned_depth_to_color/image_raw', Image)

        # 前摄像头时间同步器
        self.front_ts = message_filters.ApproximateTimeSynchronizer(
            [self.front_color_sub, self.front_depth_sub], 
            queue_size=5,
            slop=0.05
        )
        self.front_ts.registerCallback(self.front_camera_callback)
        
        # 后摄像头时间同步器
        self.rear_ts = message_filters.ApproximateTimeSynchronizer(
            [self.rear_color_sub, self.rear_depth_sub], 
            queue_size=5,
            slop=0.05
        )
        self.rear_ts.registerCallback(self.rear_camera_callback)

        # 订阅前后摄像头内参
        self.front_camera_info_sub = rospy.Subscriber('/cam_1/color/camera_info', CameraInfo, self.front_camera_info_callback)
        self.rear_camera_info_sub = rospy.Subscriber('/cam_2/color/camera_info', CameraInfo, self.rear_camera_info_callback)
        
        # 发布器
        self.camera_coords_pub = rospy.Publisher('/target_obs', PointStamped, queue_size=5)
        self.world_coords_pub = rospy.Publisher('/target_pos', PointStamped, queue_size=5)
        
        # 前摄像头内参
        self.front_fx = None
        self.front_fy = None
        self.front_cx = None
        self.front_cy = None
        self.front_camera_info_received = False
        
        # 后摄像头内参
        self.rear_fx = None
        self.rear_fy = None
        self.rear_cx = None
        self.rear_cy = None
        self.rear_camera_info_received = False

        # 订阅机械犬里程计
        self.odom_sub = rospy.Subscriber('/Odometry', Odometry, self.odom_callback)

        # 当前位姿存储
        self.current_pose = None
        self.orientation = None
        self.position = None
        
        # 处理频率控制
        self.last_process_time = rospy.Time.now()
        self.process_interval = rospy.Duration(0.02)
        
        # 图像降采样标志
        self.downsample_factor = 4
        
        # 帧跳过计数器
        self.frame_skip_counter = 0
        self.process_every_n_frames = 3
        
        # 存储两个摄像头的检测结果
        self.front_detection_result = None
        self.rear_detection_result = None
        self.detection_timestamp = rospy.Time.now()
        self.detection_timeout = rospy.Duration(0.1)  # 100ms超时

    def front_camera_callback(self, color_msg, depth_msg):
        """前摄像头回调函数"""
        # 帧跳过控制
        self.frame_skip_counter += 1
        if self.frame_skip_counter < self.process_every_n_frames:
            return
        self.frame_skip_counter = 0
        
        try:
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"前摄像头CvBridge错误: {e}")
            return
        
        # 处理前摄像头图像
        detection_result = self.detect_red_object(color_frame, depth_frame, "front")
        self.front_detection_result = detection_result
        
        # 比较并选择最佳检测结果
        self.compare_and_publish_best_detection()

    def rear_camera_callback(self, color_msg, depth_msg):
        """后摄像头回调函数"""
        try:
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
        except CvBridgeError as e:
            rospy.logerr(f"后摄像头CvBridge错误: {e}")
            return
        
        # 处理后摄像头图像
        detection_result = self.detect_red_object(color_frame, depth_frame, "rear")
        self.rear_detection_result = detection_result
        
        # 比较并选择最佳检测结果
        self.compare_and_publish_best_detection()

    def front_camera_info_callback(self, msg):
        """前摄像头内参回调"""
        if not self.front_camera_info_received:
            self.front_fx = msg.K[0]
            self.front_fy = msg.K[4]
            self.front_cx = msg.K[2]
            self.front_cy = msg.K[5]
            self.front_camera_info_received = True
            rospy.loginfo(f"接收到前摄像头内参: fx={self.front_fx}, fy={self.front_fy}, cx={self.front_cx}, cy={self.front_cy}")

    def rear_camera_info_callback(self, msg):
        """后摄像头内参回调"""
        if not self.rear_camera_info_received:
            self.rear_fx = msg.K[0]
            self.rear_fy = msg.K[4]
            self.rear_cx = msg.K[2]
            self.rear_cy = msg.K[5]
            self.rear_camera_info_received = True
            rospy.loginfo(f"接收到后摄像头内参: fx={self.rear_fx}, fy={self.rear_fy}, cx={self.rear_cx}, cy={self.rear_cy}")
    
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

    def pixel_to_3d(self, u, v, depth, camera_type):
        """将像素坐标和深度值转换为三维坐标"""
        if camera_type == "front":
            if not self.front_camera_info_received:
                rospy.logwarn("前摄像头内参尚未接收")
                return None
            fx, fy, cx, cy = self.front_fx, self.front_fy, self.front_cx, self.front_cy
        elif camera_type == "rear":
            if not self.rear_camera_info_received:
                rospy.logwarn("后摄像头内参尚未接收")
                return None
            fx, fy, cx, cy = self.rear_fx, self.rear_fy, self.rear_cx, self.rear_cy
        else:
            return None
        
        if depth <= 0:
            return None
        
        depth_m = depth / 1000.0
        
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        
        return x, y, z

    def transform_to_world(self, camera_coords, camera_type):
        """坐标转换函数，根据摄像头类型进行不同的变换"""
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

            # 根据摄像头类型调整坐标变换
            if camera_type == "front":
                # 前摄像头朝前
                world_x = robot_x + z_local * np.cos(yaw) + x_local * np.sin(yaw)
                world_y = robot_y + z_local * np.sin(yaw) - x_local * np.cos(yaw)
                world_z = robot_z
            elif camera_type == "rear":
                # 后摄像头朝后，需要旋转180度
                world_x = robot_x - z_local * np.cos(yaw) - x_local * np.sin(yaw)
                world_y = robot_y - z_local * np.sin(yaw) + x_local * np.cos(yaw)
                world_z = -robot_z
            else:
                return None
                
            # world_z = robot_z

            result = PointStamped()
            result.header = Header()
            result.header.stamp = rospy.Time.now()
            result.header.frame_id = "odom"
            result.point = Point(world_x, world_y, world_z)
            return result

        except Exception as e:
            rospy.logerr(f"Coordinate transform failed: {str(e)}")
            return None

    def detect_red_object(self, color_frame, depth_frame, camera_type):
        """在单个摄像头图像中检测红色物体"""
        # 图像降采样
        if self.downsample_factor > 1:
            height, width = color_frame.shape[:2]
            new_width = width // self.downsample_factor
            new_height = height // self.downsample_factor
            
            small_color = cv2.resize(color_frame, (new_width, new_height))
            small_depth = cv2.resize(depth_frame, (new_width, new_height))
            
            scale_x = new_width / width
            scale_y = new_height / height
        else:
            small_color = color_frame
            small_depth = depth_frame
            scale_x = scale_y = 1.0
        
        # 转换为HSV颜色空间
        hsv_img = cv2.cvtColor(small_color, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码
        mask_red = cv2.inRange(hsv_img, self.lower_red, self.upper_red)
        mask_red = cv2.medianBlur(mask_red, 5)
        
        # 查找轮廓
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 找到面积最大的红色物体
        max_area = 0
        largest_contour = None
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            min_area = 1000 // (self.downsample_factor ** 2)
            if area > min_area and area > max_area:
                max_area = area
                largest_contour = cnt
        
        detection_result = {
            'camera_type': camera_type,
            'area': max_area,
            'contour': largest_contour,
            'color_frame': color_frame,
            'depth_frame': depth_frame,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'timestamp': rospy.Time.now()
        }
        
        return detection_result

    def compare_and_publish_best_detection(self):
        """比较两个摄像头的检测结果，选择面积最大的进行发布"""
        current_time = rospy.Time.now()
        
        # 检查检测结果是否有效和及时
        valid_front = (self.front_detection_result is not None and 
                      self.front_detection_result['area'] > 0 and
                      (current_time - self.front_detection_result['timestamp']) < self.detection_timeout)
        
        valid_rear = (self.rear_detection_result is not None and 
                     self.rear_detection_result['area'] > 0 and
                     (current_time - self.rear_detection_result['timestamp']) < self.detection_timeout)
        
        best_detection = None
        
        if valid_front and valid_rear:
            # 两个摄像头都有有效检测，选择面积大的
            if self.front_detection_result['area'] >= self.rear_detection_result['area']:
                best_detection = self.front_detection_result
            else:
                best_detection = self.rear_detection_result
        elif valid_front:
            # 只有前摄像头有效检测
            best_detection = self.front_detection_result
        elif valid_rear:
            # 只有后摄像头有效检测
            best_detection = self.rear_detection_result
        
        # 处理最佳检测结果
        if best_detection is not None:
            self.process_best_detection(best_detection)

    def process_best_detection(self, detection_result):
        """处理最佳检测结果，获取深度信息并发布坐标"""
        camera_type = detection_result['camera_type']
        largest_contour = detection_result['contour']
        color_frame = detection_result['color_frame']
        depth_frame = detection_result['depth_frame']
        scale_x = detection_result['scale_x']
        scale_y = detection_result['scale_y']
        max_area = detection_result['area']
        
        if largest_contour is None:
            return
            
        (x, y, w, h) = cv2.boundingRect(largest_contour)
        
        # 如果使用了降采样，需要将坐标映射回原图
        if self.downsample_factor > 1:
            x = int(x / scale_x)
            y = int(y / scale_y)
            w = int(w / scale_x)
            h = int(h / scale_y)
        
        # 绘制检测框
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

                camera_coords_tuple = self.pixel_to_3d(center_x, center_y, depth_value_mm, camera_type)
                if camera_coords_tuple:
                    x_cam, y_cam, z_cam = camera_coords_tuple
                    
                    # 发布相机坐标系坐标
                    camera_point_stamped = PointStamped()
                    camera_point_stamped.header.stamp = rospy.Time.now()
                    camera_point_stamped.header.frame_id = f"{camera_type}_camera_color_optical_frame"
                    # 根据摄像头类型设置坐标
                    if camera_type == "front":
                        camera_point_stamped.point = Point(x_cam, y_cam, z_cam)
                    elif camera_type == "rear":
                        camera_point_stamped.point = Point(x_cam, y_cam, -z_cam)
                    self.camera_coords_pub.publish(camera_point_stamped)
                    
                    if self.num % 10 == 0:
                        rospy.loginfo(f"[{camera_type}摄像头] 目标在相机坐标系下: x={x_cam:.2f}, y={y_cam:.2f}, z={z_cam:.2f}")

                    # 转换到世界坐标系并发布
                    camera_coords = [x_cam, y_cam, z_cam]
                    world_point_stamped = self.transform_to_world(camera_coords, camera_type)
                    if world_point_stamped:
                        self.world_coords_pub.publish(world_point_stamped)
                        if self.num % 10 == 0:
                            rospy.loginfo(f"[{camera_type}摄像头] 目标在世界坐标系下: x={world_point_stamped.point.x:.2f}, y={world_point_stamped.point.y:.2f}, z={world_point_stamped.point.z:.2f}")
            else:
                depth_text = "Depth: 0mm (Invalid)"
        else:
            depth_text = "Depth: OOB"

        # 添加摄像头标识和检测信息
        cv2.putText(color_frame, f"{camera_type.upper()} - Red (Area: {int(max_area * (self.downsample_factor ** 2))})", 
                   (x, y - 25), self.font, 0.7, (0, 0, 255), 2)
        cv2.putText(color_frame, depth_text, (x, y - 5), self.font, 0.7, (0, 255, 0), 2)
        
        self.num += 1
        
        # 显示图像（可选）
        # cv2.imshow(f"{camera_type} Camera Detection", color_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            rospy.signal_shutdown("用户按下ESC键退出")

    def cleanup(self):
        """清理资源"""
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('multi_camera_red_detection_node', anonymous=True)
    
    try:
        detector = MultiCameraColorDetection()
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("多摄像头红色检测节点被中断")
    except KeyboardInterrupt:
        rospy.loginfo("收到键盘中断信号")
    finally:
        cv2.destroyAllWindows()
        rospy.loginfo("多摄像头红色检测节点已关闭")