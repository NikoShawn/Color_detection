import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from geometry_msgs.msg import PointStamped,TransformStamped,Point
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Header  # 添加这行导入

class DepthOfRed:
    def __init__(self):
        self.bridge = CvBridge()
        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        
        # 订阅相机内参
        self.camera_info_sub = rospy.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo, self.camera_info_callback)
        
        # 添加发布器
        self.camera_coords_pub = rospy.Publisher('/target_obs', PointStamped, queue_size=10)
        self.world_coords_pub = rospy.Publisher('/target_pos', PointStamped, queue_size=10)
        
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

        # 等待首次里程计数据
        # rospy.loginfo("Waiting for initial odometry...")
        # while not rospy.is_shutdown() and self.current_pose is None:
        #     rospy.sleep(0.1)
        # rospy.loginfo("Odometry ready!")

        # 使用 ApproximateTimeSynchronizer 来同步具有相似时间戳的消息
        # 调整 slop 参数（以秒为单位）以允许时间戳之间的一些差异
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.2  # 0.1 秒的容差
        )
        self.ts.registerCallback(self.callback)
        # rospy.loginfo("订阅了颜色和深度图像话题，并同步它们。")

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

    def callback(self, color_msg, depth_msg):
        rospy.loginfo("接收到同步的图像消息。")
        
        if not self.camera_info_received:
            rospy.logwarn("等待相机内参...")
            return
            
        try:
            # 将 ROS Image 消息转换为 OpenCV 图像
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough") # 通常是 16UC1 或 32FC1
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge 错误: {e}")
            return

        # 定义红色的 HSV 范围
        # 这些值可能需要根据您的具体照明条件和相机进行调整
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])

        # 将 BGR 图像转换为 HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # 创建红色区域的掩码，只使用mask1
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        
        # 对mask进行滤波处理，去除杂点
        # 1. 形态学操作：先开运算（去除小噪点），再闭运算（填充小洞）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=2)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # 2. 高斯模糊
        mask1 = cv2.GaussianBlur(mask1, (5, 5), 0)
        
        # 3. 重新二值化
        _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
        
        # 4. 使用连通组件分析，仅当最大区域的面积大于等于阈值时，才保留该最大区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask1, connectivity=8)
        min_area = 1000  # 最小面积阈值，可以根据需要调整
        
        output_mask = np.zeros_like(mask1) # 初始化一个空掩码

        if num_labels > 1:  # 确保至少有一个前景组件 (标签0是背景)
            # stats[0] 对应背景。我们关注前景组件 stats[1:]
            # 获取所有前景组件的面积
            foreground_areas = stats[1:, cv2.CC_STAT_AREA]
            
            # 找到前景组件中的最大面积
            max_foreground_area = np.max(foreground_areas)
            
            # 如果这个最大面积大于等于设定的最小面积阈值
            if max_foreground_area >= min_area:
                # 找到最大面积对应的组件的标签
                # np.argmax(foreground_areas) 返回在 foreground_areas 中的索引
                # 由于 foreground_areas 是从 stats[1:] 开始的，所以实际标签值是索引 + 1
                largest_component_label = np.argmax(foreground_areas) + 1
                
                # 在输出掩码中只保留这个最大的且符合条件的区域
                output_mask[labels == largest_component_label] = 255
        
        mask1 = output_mask # 更新 mask1

        # 查找红色像素的坐标
        red_pixel_coordinates = np.argwhere(mask1 > 0) # 返回 (row, col) 形式的索引

        if red_pixel_coordinates.size > 0:
            # rospy.loginfo(f"检测到 {len(red_pixel_coordinates)} 个红色像素。")
            valid_depths = []
            valid_3d_points = []
            
            for coord in red_pixel_coordinates:
                row, col = coord
                # 获取对应像素的深度值
                # 深度图像的单位通常是毫米 (对于 16UC1) 或米 (对于 32FC1)
                # 请查阅您的相机文档以确认单位
                depth_value = depth_image[row, col]
                
                # 仅处理有效的深度值（例如，非零值，具体取决于您的深度传感器）
                if depth_value > 0: 
                    valid_depths.append(depth_value)
                    
                    # 计算三维坐标
                    point_3d = self.pixel_to_3d(col, row, depth_value)
                    if point_3d is not None:
                        x, y, z = point_3d
                        valid_3d_points.append((x, y, z))
                        # rospy.loginfo(f"红色像素位于 ({col}, {row})，深度: {depth_value}mm，三维坐标: ({x:.3f}, {y:.3f}, {z:.3f})m")
            
            if valid_depths:
                average_depth = np.mean(valid_depths)
                # rospy.loginfo(f"检测到的红色区域的平均深度: {average_depth}mm")
                
                if valid_3d_points:
                    # 计算平均三维坐标
                    avg_x = np.mean([p[0] for p in valid_3d_points])
                    avg_y = np.mean([p[1] for p in valid_3d_points])
                    avg_z = np.mean([p[2] for p in valid_3d_points])
                    
                    # 保存到camera_coords
                    self.camera_coords = [avg_x, avg_y, avg_z]
                    
                    # rospy.loginfo(f"红色区域的平均三维坐标 (相机光学坐标系): x={avg_x:.3f}m, y={avg_y:.3f}m, z={avg_z:.3f}m")
            else:
                rospy.loginfo("检测到红色像素，但没有有效的深度值。")
        else:
            rospy.loginfo("未检测到红色像素。")
            # 如果没有检测到红色像素，可以选择清空camera_coords
            self.camera_coords = None

        # （可选）显示图像以进行调试
        cv2.imshow("Color Image", color_image)
        # cv2.imshow("Depth Image", depth_image) # 归一化以便显示
        # cv2.imshow("Red Mask", mask1)
        cv2.waitKey(1)

    def get_camera_coords(self):
        """获取最新的相机坐标"""
        return self.camera_coords

if __name__ == '__main__':
    # 只初始化一次节点
    rospy.init_node('red_object_world_locator', anonymous=True)
    
    # 创建实例
    dor = DepthOfRed()
    
    # 创建一个定时器来定期检查和打印坐标
    def print_coordinates():
        camera_coords = dor.get_camera_coords()
        if camera_coords is not None:
            # 创建相机坐标系的PointStamped消息
            camera_point = PointStamped()
            camera_point.header = Header()
            camera_point.header.stamp = rospy.Time.now()
            camera_point.header.frame_id = "camera_optical_frame"
            camera_point.point = Point(camera_coords[0], camera_coords[1], camera_coords[2])
            
            # 发布相机坐标
            dor.camera_coords_pub.publish(camera_point)
            
            world_point = dor.transform_to_world(camera_coords)
            # rospy.loginfo(f"相机坐标系下的点: {camera_coords}")
            if world_point is not None:
                # 发布世界坐标
                dor.world_coords_pub.publish(world_point)
                # rospy.loginfo(f"世界坐标系下的点: ({world_point.point.x:.3f}, {world_point.point.y:.3f}, {world_point.point.z:.3f})")
        else:
            rospy.loginfo("暂无检测到的红色物体")
    
    # 创建定时器，每秒打印一次坐标信息
    timer = rospy.Timer(rospy.Duration(0.05), lambda event: print_coordinates())
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("关闭节点。")
        timer.shutdown()
    # cv2.destroyAllWindows() # 如果使用了 cv2.imshow