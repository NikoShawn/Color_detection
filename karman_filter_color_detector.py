#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters # 新增导入
from geometry_msgs.msg import PointStamped,TransformStamped,Point
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Header  # 添加这行导入

class KalmanFilter:
    """卡尔曼滤波器类，用于滤波3D坐标"""
    def __init__(self, dt=1.0):
        # 状态向量: [x, y, z, vx, vy, vz] (位置 + 速度)
        self.state = np.zeros((6, 1))  # 状态向量
        
        # 状态转移矩阵 (匀速模型)
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 观测矩阵 (只能观测位置)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # 过程噪声协方差矩阵
        q = 0.1  # 过程噪声标准差
        self.Q = np.eye(6) * q**2
        
        # 观测噪声协方差矩阵
        r = 0.3  # 观测噪声标准差 (根据深度相机精度调整)
        self.R = np.eye(3) * r**2
        
        # 误差协方差矩阵 (初始不确定性)
        self.P = np.eye(6) * 10.0
        
        # 是否已初始化
        self.initialized = False
    
    def predict(self):
        """预测步骤"""
        # 预测状态
        self.state = self.F @ self.state
        
        # 预测误差协方差
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        """更新步骤"""
        measurement = np.array(measurement).reshape(3, 1)
        
        if not self.initialized:
            # 首次初始化状态
            self.state[:3] = measurement
            self.state[3:] = 0  # 初始速度为0
            self.initialized = True
            return self.state[:3].flatten()
        
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        innovation = measurement - self.H @ self.state
        self.state = self.state + K @ innovation
        
        # 更新误差协方差
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state[:3].flatten()  # 返回滤波后的位置
    
    def get_filtered_position(self):
        """获取当前滤波后的位置"""
        return self.state[:3].flatten()

class ColorDetection:
    def __init__(self):
        self.bridge = CvBridge()
        
        # 红色范围定义
        self.lower_red = np.array([165, 100, 100])     # 红色范围低阈值
        self.upper_red = np.array([180, 255, 255])    # 红色范围高阈值

        # self.lower_red = np.array([85, 100, 100])     # 红色范围低阈值
        # self.upper_red = np.array([105, 255, 255])    # 红色范围高阈值
        
        # 深蓝色范围选项2 - 包含海军蓝
        self.lower_blue = np.array([100, 100, 30])     
        self.upper_blue = np.array([125, 255, 120])  
        
        # 字体设置
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 图片保存计数器
        self.num = 0
        
        # 初始化卡尔曼滤波器
        self.kalman_red = KalmanFilter(dt=0.1)  # 假设10Hz更新频率
        self.kalman_blue = KalmanFilter(dt=0.1)
        
        # 订阅颜色图像和深度图像话题
        # self.color_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback) # 旧的订阅方式
        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        # 假设深度相机话题为 /camera/depth/image_rect_raw，请根据实际情况修改
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image) 

        # 使用 ApproximateTimeSynchronizer 同步颜色和深度图像消息
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1 # 允许消息之间最大0.1秒的时间差
        )
        self.ts.registerCallback(self.synchronized_callback) # 注册新的同步回调函数

        # 订阅相机内参
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)
        
        # 添加发布器 - 红色目标
        self.camera_coords_pub = rospy.Publisher('/target_obs', PointStamped, queue_size=10)
        self.world_coords_pub = rospy.Publisher('/target_pos', PointStamped, queue_size=10)
        # 添加发布器 - 蓝色目标
        self.blue_camera_coords_pub = rospy.Publisher('/teamate_obs', PointStamped, queue_size=10)
        self.blue_world_coords_pub = rospy.Publisher('/teamate_pos', PointStamped, queue_size=10)
        # 新增：发布滤波后的坐标
        self.filtered_camera_coords_pub = rospy.Publisher('/target_obs_filtered', PointStamped, queue_size=10)
        self.filtered_world_coords_pub = rospy.Publisher('/target_pos_filtered', PointStamped, queue_size=10)
        self.blue_filtered_camera_coords_pub = rospy.Publisher('/teamate_obs_filtered', PointStamped, queue_size=10)
        self.blue_filtered_world_coords_pub = rospy.Publisher('/teamate_pos_filtered', PointStamped, queue_size=10)
        # 新增：发布处理后的彩色图像
        self.processed_image_pub = rospy.Publisher('/processed_color_image', Image, queue_size=10)
        
        # 相机内参变量
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_info_received = False
        
        # 添加相机坐标存储变量
        self.camera_coords = None
        self.blue_camera_coords = None

        # 订阅机械犬里程计
        self.odom_sub = rospy.Subscriber('/Odometry', Odometry, self.odom_callback)

        # rospy.loginfo("Waiting for initial odometry...")
        # while not rospy.is_shutdown() and self.current_pose is None:
        #     rospy.sleep(0.1)
        # rospy.loginfo("Odometry ready!")

        # 当前位姿存储
        self.current_pose = None
        self.orientation = None
        self.position = None

    def synchronized_callback(self, color_msg, depth_msg):
        """处理接收到的同步颜色和深度图像消息"""
        try:
            # 将ROS Image消息转换为OpenCV图像
            color_frame = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            # 深度图像通常是16位无符号整数 (16UC1)，单位为毫米
            # 或32位浮点数 (32FC1)，单位为米。请根据您的相机输出调整 "16UC1"
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1") 
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge错误: {e}")
            return
        
        # 红色检测和深度处理
        self.process_frame_and_depth(color_frame, depth_frame)

    def camera_info_callback(self, msg):
        """处理相机内参回调"""
        if not self.camera_info_received:
            # 提取相机内参矩阵的参数
            # K = [fx  0 cx]
            #     [ 0 fy cy]
            #     [ 0  0  1]
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
        """
        :param camera_coords: 机械犬基座坐标系下的三维坐标 [x,y,z] (list/numpy array)
        :return: 世界坐标系下的PointStamped消息
        """
        if self.current_pose is None:
            rospy.logwarn("No odometry data received yet!")
            return None

        try:
            # 提取机器人位姿
            robot_x = self.position[0]
            robot_y = self.position[1]
            robot_z = self.position[2]

            # 从四元数提取偏航角(yaw)
            q = self.orientation
            roll, pitch, yaw = euler_from_quaternion([q[0], q[1], q[2], q[3]])

            # 解析局部坐标（假设x为左侧，z为前方）
            x_local = camera_coords[0]  # 左侧偏移
            z_local = camera_coords[2]  # 前方距离
            #y_local = camera_coords[1]  # 高度

            # 执行二维坐标变换
            world_x = robot_x + z_local * np.cos(yaw) + x_local * np.sin(yaw)
            world_y = robot_y + z_local * np.sin(yaw) - x_local * np.cos(yaw)
            world_z = robot_z

            # 封装为PointStamped
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
        
        # 深度值通常以毫米为单位，转换为米
        depth_m = depth / 1000.0
        
        # 计算三维坐标 (相机光学坐标系)
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        
        return x, y, z

    def process_frame_and_depth(self, color_frame, depth_frame): # 修改函数名并增加 depth_frame 参数
        """处理单帧图像进行红色和蓝色检测并获取深度信息"""
        # 执行卡尔曼滤波器的预测步骤
        self.kalman_red.predict()
        self.kalman_blue.predict()
        
        # 转换为HSV颜色空间
        hsv_img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        
        # 根据红色范围创建掩码
        mask_red = cv2.inRange(hsv_img, self.lower_red, self.upper_red)
        # 根据蓝色范围创建掩码
        mask_blue = cv2.inRange(hsv_img, self.lower_blue, self.upper_blue)
        
        # 中值滤波去噪
        mask_red = cv2.medianBlur(mask_red, 7)
        mask_blue = cv2.medianBlur(mask_blue, 7)
        
        # 查找红色轮廓
        contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 查找蓝色轮廓
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # 处理红色检测
        max_red_area = 0
        largest_red_contour = None
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if area > 100 and area > max_red_area:  # 过滤掉太小的区域并找到最大面积
                max_red_area = area
                largest_red_contour = cnt
        
        # 只绘制面积最大的红色物体检测框并获取深度
        if largest_red_contour is not None:
            (x, y, w, h) = cv2.boundingRect(largest_red_contour)
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            
            # 计算检测框中心点
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(color_frame, (center_x, center_y), 5, (255, 0, 0), -1) # 绘制中心点

            depth_value_mm = 0
            depth_text = "Depth: N/A"
            # 确保中心点在深度图像的有效范围内
            if 0 <= center_y < depth_frame.shape[0] and 0 <= center_x < depth_frame.shape[1]:
                depth_value_mm = depth_frame[center_y, center_x]
                
                # 检查深度值是否有效（通常0表示无效深度）
                if depth_value_mm > 0:
                    depth_value_m = depth_value_mm / 1000.0  # 转换为米
                    depth_text = f"Depth: {depth_value_m:.2f}m"
                    rospy.loginfo(f"检测到红色物体，深度值: {depth_value_mm}mm ({depth_value_m:.2f}m)")
                    # 调用pixel_to_3d获取相机坐标
                    camera_coords_tuple = self.pixel_to_3d(center_x, center_y, depth_value_mm)
                    if camera_coords_tuple:
                        x_cam, y_cam, z_cam = camera_coords_tuple
                        
                        # 发布原始相机坐标
                        camera_point_stamped = PointStamped()
                        camera_point_stamped.header.stamp = rospy.Time.now()
                        # 重要: 确保这个frame_id与您的相机光学坐标系frame_id一致
                        # 通常可能是 'camera_color_optical_frame', 'camera_depth_optical_frame' 或类似名称
                        camera_point_stamped.header.frame_id = "camera_color_optical_frame" 
                        camera_point_stamped.point = Point(x_cam, y_cam, z_cam)
                        self.camera_coords_pub.publish(camera_point_stamped)
                        
                        # 应用卡尔曼滤波
                        filtered_coords = self.kalman_red.update([x_cam, y_cam, z_cam])
                        
                        # 发布滤波后的相机坐标
                        filtered_camera_point_stamped = PointStamped()
                        filtered_camera_point_stamped.header.stamp = rospy.Time.now()
                        filtered_camera_point_stamped.header.frame_id = "camera_color_optical_frame"
                        filtered_camera_point_stamped.point = Point(filtered_coords[0], filtered_coords[1], filtered_coords[2])
                        self.filtered_camera_coords_pub.publish(filtered_camera_point_stamped)
                        
                        rospy.loginfo(f"红色目标原始坐标: x={x_cam:.2f}, y={y_cam:.2f}, z={z_cam:.2f}")
                        rospy.loginfo(f"红色目标滤波后: x={filtered_coords[0]:.2f}, y={filtered_coords[1]:.2f}, z={filtered_coords[2]:.2f}")

                        # 将原始相机坐标转换为列表，用于transform_to_world
                        self.camera_coords = [x_cam, y_cam, z_cam] 

                        # 调用transform_to_world获取原始世界坐标
                        world_point_stamped = self.transform_to_world(self.camera_coords)
                        if world_point_stamped:
                            self.world_coords_pub.publish(world_point_stamped)
                        
                        # 调用transform_to_world获取滤波后的世界坐标
                        filtered_world_point_stamped = self.transform_to_world(filtered_coords)
                        if filtered_world_point_stamped:
                            self.filtered_world_coords_pub.publish(filtered_world_point_stamped)
                        else:
                            rospy.logwarn("无法将滤波后的相机坐标转换为世界坐标")
                    else:
                        rospy.logwarn("无法从像素坐标计算相机坐标")
                else:
                    depth_text = "Depth: 0mm (Invalid)"
            else:
                depth_text = "Depth: OOB" # Out of Bounds

            cv2.putText(color_frame, f"Red (Area: {int(max_red_area)})", (x, y - 25), self.font, 0.7, (0, 0, 255), 2)
            cv2.putText(color_frame, depth_text, (x, y - 5), self.font, 0.7, (0, 255, 0), 2) # 显示深度信息
        
        # 处理蓝色检测
        max_blue_area = 0
        largest_blue_contour = None
        
        for cnt in contours_blue:
            area = cv2.contourArea(cnt)
            if area > 100 and area > max_blue_area:  # 过滤掉太小的区域并找到最大面积
                max_blue_area = area
                largest_blue_contour = cnt
        
        # 只绘制面积最大的蓝色物体检测框并获取深度
        if largest_blue_contour is not None:
            (x, y, w, h) = cv2.boundingRect(largest_blue_contour)
            cv2.rectangle(color_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 青色框
            
            # 计算检测框中心点
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(color_frame, (center_x, center_y), 5, (255, 0, 255), -1)  # 紫色中心点

            depth_value_mm = 0 
            depth_text = "Depth: N/A"
            # 确保中心点在深度图像的有效范围内
            if 0 <= center_y < depth_frame.shape[0] and 0 <= center_x < depth_frame.shape[1]:
                depth_value_mm = depth_frame[center_y, center_x]
                
                # 检查深度值是否有效（通常0表示无效深度）
                if depth_value_mm > 0:
                    depth_value_m = depth_value_mm / 1000.0  # 转换为米
                    depth_text = f"Depth: {depth_value_m:.2f}m"
                    rospy.loginfo(f"检测到蓝色物体，深度值: {depth_value_mm}mm ({depth_value_m:.2f}m)")
                    # 调用pixel_to_3d获取相机坐标
                    blue_camera_coords_tuple = self.pixel_to_3d(center_x, center_y, depth_value_mm)
                    if blue_camera_coords_tuple:
                        x_cam, y_cam, z_cam = blue_camera_coords_tuple
                        
                        # 发布蓝色目标的原始相机坐标
                        blue_camera_point_stamped = PointStamped()
                        blue_camera_point_stamped.header.stamp = rospy.Time.now()
                        blue_camera_point_stamped.header.frame_id = "camera_color_optical_frame" 
                        blue_camera_point_stamped.point = Point(x_cam, y_cam, z_cam)
                        self.blue_camera_coords_pub.publish(blue_camera_point_stamped)
                        
                        # 应用卡尔曼滤波
                        blue_filtered_coords = self.kalman_blue.update([x_cam, y_cam, z_cam])
                        
                        # 发布滤波后的蓝色目标相机坐标
                        blue_filtered_camera_point_stamped = PointStamped()
                        blue_filtered_camera_point_stamped.header.stamp = rospy.Time.now()
                        blue_filtered_camera_point_stamped.header.frame_id = "camera_color_optical_frame"
                        blue_filtered_camera_point_stamped.point = Point(blue_filtered_coords[0], blue_filtered_coords[1], blue_filtered_coords[2])
                        self.blue_filtered_camera_coords_pub.publish(blue_filtered_camera_point_stamped)
                        
                        rospy.loginfo(f"蓝色目标原始坐标: x={x_cam:.2f}, y={y_cam:.2f}, z={z_cam:.2f}")
                        rospy.loginfo(f"蓝色目标滤波后: x={blue_filtered_coords[0]:.2f}, y={blue_filtered_coords[1]:.2f}, z={blue_filtered_coords[2]:.2f}")

                        # 将相机坐标转换为列表，用于transform_to_world
                        self.blue_camera_coords = [x_cam, y_cam, z_cam] 

                        # 调用transform_to_world获取原始世界坐标
                        blue_world_point_stamped = self.transform_to_world(self.blue_camera_coords)
                        if blue_world_point_stamped:
                            self.blue_world_coords_pub.publish(blue_world_point_stamped)
                        
                        # 调用transform_to_world获取滤波后的世界坐标
                        blue_filtered_world_point_stamped = self.transform_to_world(blue_filtered_coords)
                        if blue_filtered_world_point_stamped:
                            self.blue_filtered_world_coords_pub.publish(blue_filtered_world_point_stamped)
                        else:
                            rospy.logwarn("无法将蓝色目标的滤波后相机坐标转换为世界坐标")
                    else:
                        rospy.logwarn("无法从蓝色目标的像素坐标计算相机坐标")
                else:
                    depth_text = "Depth: 0mm (Invalid)"
            else:
                depth_text = "Depth: OOB" # Out of Bounds

            cv2.putText(color_frame, f"Blue (Area: {int(max_blue_area)})", (x, y - 25), self.font, 0.7, (255, 0, 0), 2)
            cv2.putText(color_frame, depth_text, (x, y - 5), self.font, 0.7, (0, 255, 255), 2) # 显示深度信息
            
        # 新增：发布处理后的彩色图像作为ROS话题
        try:
            processed_image_msg = self.bridge.cv2_to_imgmsg(color_frame, "bgr8")
            processed_image_msg.header.stamp = rospy.Time.now()
            processed_image_msg.header.frame_id = "camera_color_optical_frame"
            self.processed_image_pub.publish(processed_image_msg)
        except CvBridgeError as e:
            rospy.logerr(f"发布处理后图像时出错: {e}")
        
        # 更新图片计数器
        self.num += 1
        
        # 显示结果
        # cv2.imshow("Red Object Detection", color_frame)
        
        # 保存图片（可选，创建imgs文件夹）
        # try:
        #     cv2.imwrite(f"imgs/{self.num}.jpg", color_frame)
        #     rospy.loginfo(f"保存图片: imgs/{self.num}.jpg")
        # except:
        #     pass  # 如果imgs文件夹不存在，跳过保存
        
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
        camera_coords = detector
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