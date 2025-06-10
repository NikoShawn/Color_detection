#!/usr/bin/env python

import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Imu
from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header
import tf.transformations
import time
from scipy.spatial.distance import cdist

# 自定义消息类型（如果没有包，使用字典形式发布）
class PersonTrackMsg:
    def __init__(self):
        self.header = Header()
        self.track_id = 0
        self.position = Point()
        self.velocity = Vector3()
        self.speed = 0.0
        self.age = 0
        self.hits = 0
        self.is_valid = False

class KalmanFilter:
    """3D位置跟踪的卡尔曼滤波器"""
    def __init__(self, dt=0.1):
        self.dt = dt
        # 状态向量：[x, y, z, vx, vy, vz] (位置和速度)
        self.x = np.zeros(6)
        
        # 状态转移矩阵
        self.F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 观测矩阵（只观测位置）
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # 过程噪声协方差
        self.Q = np.eye(6) * 0.1
        
        # 观测噪声协方差
        self.R = np.eye(3) * 0.5
        
        # 误差协方差矩阵
        self.P = np.eye(6) * 1.0
        
        self.is_initialized = False
        self.last_update_time = None
    
    def predict(self):
        """预测步骤"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        """更新步骤"""
        if not self.is_initialized:
            # 首次初始化
            self.x[:3] = measurement
            self.x[3:] = 0  # 初始速度为0
            self.is_initialized = True
            self.last_update_time = rospy.Time.now()
            return
        
        # 更新时间间隔
        current_time = rospy.Time.now()
        if self.last_update_time is not None:
            self.dt = (current_time - self.last_update_time).to_sec()
            self.dt = max(0.01, min(0.5, self.dt))  # 限制dt范围
            # 更新状态转移矩阵
            self.F[0, 3] = self.dt
            self.F[1, 4] = self.dt
            self.F[2, 5] = self.dt
        
        self.last_update_time = current_time
        
        # 预测
        self.predict()
        
        # 计算卡尔曼增益
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        y = measurement - self.H @ self.x  # 残差
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
    
    def get_position(self):
        """获取预测位置"""
        return self.x[:3]
    
    def get_velocity(self):
        """获取预测速度"""
        return self.x[3:]

class PersonTracker:
    """人体跟踪器"""
    def __init__(self, track_id, initial_position):
        self.track_id = track_id
        self.kalman = KalmanFilter()
        self.kalman.update(initial_position)
        self.last_seen = rospy.Time.now()
        self.consecutive_misses = 0
        self.age = 0
        self.hits = 1
    
    def predict(self):
        """预测下一个位置"""
        self.kalman.predict()
        return self.kalman.get_position()
    
    def update(self, position):
        """用新的观测更新跟踪"""
        self.kalman.update(position)
        self.last_seen = rospy.Time.now()
        self.consecutive_misses = 0
        self.hits += 1
        self.age += 1
    
    def miss(self):
        """标记一次错过"""
        self.consecutive_misses += 1
        self.age += 1
    
    def is_valid(self):
        """判断跟踪是否有效"""
        # 需要至少3次命中，且连续错过次数少于5次
        return self.hits >= 3 and self.consecutive_misses < 5
    
    def should_delete(self):
        """判断是否应该删除跟踪"""
        time_since_last_seen = (rospy.Time.now() - self.last_seen).to_sec()
        return self.consecutive_misses > 10 or time_since_last_seen > 2.0

class PersonDetector:
    def __init__(self):
        rospy.init_node('person_detector', anonymous=True)
        
        # 参数设置
        self.min_points = rospy.get_param('~min_points', 30)
        self.max_points = rospy.get_param('~max_points', 5000)
        self.eps = rospy.get_param('~eps', 0.1)
        self.min_samples = rospy.get_param('~min_samples', 5)
        self.angle_range = rospy.get_param('~angle_range', 60)
        
        # 跟踪参数
        self.max_distance = rospy.get_param('~max_distance', 1.0)  # 关联距离阈值
        
        # 订阅话题
        self.lidar_sub = rospy.Subscriber('/livox/lidar', PointCloud2, self.lidar_callback)
        self.imu_sub = rospy.Subscriber('/livox/imu', Imu, self.imu_callback)
        
        # 发布话题
        self.marker_pub = rospy.Publisher('/detected_persons', MarkerArray, queue_size=1)
        self.cluster_pub = rospy.Publisher('/person_clusters', PointCloud2, queue_size=1)
        self.person_points_pub = rospy.Publisher('/person_points', PointCloud2, queue_size=1)
        self.tracks_pub = rospy.Publisher('/tracked_persons', MarkerArray, queue_size=1)
        
        self.imu_data = None
        self.yaw = 0.0
        
        # 跟踪器相关
        self.trackers = []
        
        # 数据记录
        self.tracking_data = {}
        
        rospy.loginfo("人体检测节点已初始化，等待LiDAR和IMU数据...")
    
    def imu_callback(self, msg):
        """处理IMU数据并提取朝向信息"""
        self.imu_data = msg
        
        # 从IMU的四元数中提取偏航角(yaw)
        quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.yaw = euler[2]
    
    def filter_point_cloud(self, points):
        """过滤点云数据"""
        # 提取x, y, z坐标
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # 距离过滤（小于30米）
        dist = np.sqrt(x**2 + y**2 + z**2)
        dist_mask = dist < 30.0
        
        # 高度过滤（假设地面以上0.2m到2.2m之间可能是人）
        height_mask = (z > 0.2) & (z < 1.8)
        
        # 角度过滤 (IMU朝向左右各60度范围)
        angles = np.arctan2(y, x)
        angle_diffs = np.abs((angles - self.yaw + np.pi) % (2 * np.pi) - np.pi)
        angle_threshold = np.radians(self.angle_range)
        angle_mask = angle_diffs <= angle_threshold
        
        # 结合所有过滤条件
        mask = dist_mask & height_mask & angle_mask
        
        return points[mask]
    
    def detect_persons(self, points):
        """使用DBSCAN识别人体，最多识别2个人"""
        # 仅使用空间坐标(x,y,z)进行聚类
        X = points[:, :3]
        
        # 应用DBSCAN
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(X)
        labels = db.labels_
        
        # 收集符合人体特征的簇
        person_clusters = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            # 忽略噪声点（标签为-1）
            if label == -1:
                continue
            
            # 如果已经检测到2个人，停止处理更多的簇
            if len(person_clusters) >= 2:
                break
                
            # 获取当前簇的所有点
            cluster_points = points[labels == label]
            
            # 判断簇的大小是否符合人体特征
            if self.min_points <= len(cluster_points) <= self.max_points:
                # 计算簇的几何特征
                min_z = np.min(cluster_points[:, 2])
                max_z = np.max(cluster_points[:, 2])
                height = max_z - min_z
                
                # 判断高度是否符合人体特征（0.8m到2.0m之间）
                if 0.8 <= height <= 2.0:
                    person_clusters.append(cluster_points)
        
        # 如果检测到超过2个人，选择最大的2个簇
        if len(person_clusters) > 2:
            # 按点的数量排序，选择最大的2个
            person_clusters.sort(key=len, reverse=True)
            person_clusters = person_clusters[:2]
        
        return person_clusters
    
    def associate_detections_to_tracks(self, detections):
        """将检测结果关联到现有轨迹"""
        if not self.trackers or not detections:
            return [], list(range(len(detections))), []
        
        # 获取所有跟踪器的预测位置
        track_positions = np.array([tracker.predict() for tracker in self.trackers])
        
        # 计算检测点与轨迹的距离矩阵
        detection_positions = np.array([np.mean(det[:, :3], axis=0) for det in detections])
        distances = cdist(detection_positions, track_positions)
        
        # 匈牙利算法进行关联（简化版本）
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.trackers)))
        
        # 贪心匹配
        while distances.size > 0:
            min_idx = np.unravel_index(np.argmin(distances), distances.shape)
            det_idx, track_idx = min_idx
            
            if distances[det_idx, track_idx] < self.max_distance:
                # 匹配成功
                real_det_idx = unmatched_detections[det_idx]
                real_track_idx = unmatched_tracks[track_idx]
                matched_indices.append((real_det_idx, real_track_idx))
                
                # 从未匹配列表中移除
                unmatched_detections.remove(real_det_idx)
                unmatched_tracks.remove(real_track_idx)
                
                # 从距离矩阵中移除对应行列
                distances = np.delete(distances, det_idx, axis=0)
                distances = np.delete(distances, track_idx, axis=1)
            else:
                break
        
        return matched_indices, unmatched_detections, unmatched_tracks
    
    def reassign_track_ids(self):
        """重新分配Track ID，使其表示当前人数，最多2个人"""
        valid_trackers = [t for t in self.trackers if t.is_valid()]
        # 限制最多2个有效跟踪器
        valid_trackers = valid_trackers[:2]
        
        for i, tracker in enumerate(valid_trackers):
            tracker.track_id = i + 1  # ID从1开始，最多到2
    
    def update_trackers(self, detections):
        """更新跟踪器，最多跟踪2个人"""
        # 关联检测结果到轨迹
        matched, unmatched_dets, unmatched_tracks = self.associate_detections_to_tracks(detections)
        
        # 更新匹配的轨迹
        for det_idx, track_idx in matched:
            centroid = np.mean(detections[det_idx][:, :3], axis=0)
            self.trackers[track_idx].update(centroid)
        
        # 标记未匹配的轨迹为错过
        for track_idx in unmatched_tracks:
            self.trackers[track_idx].miss()
        
        # 为未匹配的检测创建新轨迹，但最多只能有2个跟踪器
        current_valid_trackers = len([t for t in self.trackers if not t.should_delete()])
        
        for det_idx in unmatched_dets:
            if current_valid_trackers >= 2:
                break  # 已经达到最大跟踪数量，停止创建新轨迹
            
            centroid = np.mean(detections[det_idx][:, :3], axis=0)
            new_tracker = PersonTracker(0, centroid)  # 临时ID
            self.trackers.append(new_tracker)
            current_valid_trackers += 1
        
        # 移除应该删除的轨迹
        self.trackers = [t for t in self.trackers if not t.should_delete()]
        
        # 如果跟踪器数量超过2个，保留最稳定的2个
        valid_trackers = [t for t in self.trackers if t.is_valid()]
        if len(valid_trackers) > 2:
            # 按hits数量排序，保留最稳定的2个
            valid_trackers.sort(key=lambda x: x.hits, reverse=True)
            # 移除多余的跟踪器
            trackers_to_remove = valid_trackers[2:]
            for tracker in trackers_to_remove:
                if tracker in self.trackers:
                    self.trackers.remove(tracker)
        
        # 重新分配ID，使其表示当前人数
        self.reassign_track_ids()
    
    def record_tracking_data(self):
        """记录跟踪数据"""
        current_time = rospy.Time.now()
        
        for tracker in self.trackers:
            if tracker.is_valid():
                position = tracker.kalman.get_position()
                velocity = tracker.kalman.get_velocity()
                speed = np.linalg.norm(velocity)
                
                # 更新跟踪数据
                self.tracking_data[tracker.track_id] = {
                    'timestamp': current_time.to_sec(),
                    'position': {
                        'x': float(position[0]),
                        'y': float(position[1]),
                        'z': float(position[2])
                    },
                    'velocity': {
                        'x': float(velocity[0]),
                        'y': float(velocity[1]),
                        'z': float(velocity[2])
                    },
                    'speed': float(speed),
                    'age': tracker.age,
                    'hits': tracker.hits,
                    'is_valid': tracker.is_valid()
                }
    
    def publish_tracking_data(self):
        """发布跟踪数据"""
        # 准备发布数据
        valid_trackers = [t for t in self.trackers if t.is_valid()]
        
        # 打印跟踪信息到控制台
        if valid_trackers:
            rospy.loginfo("当前检测到 %d 个人:", len(valid_trackers))
            for tracker in valid_trackers:
                position = tracker.kalman.get_position()
                velocity = tracker.kalman.get_velocity()
                speed = np.linalg.norm(velocity)
                
                rospy.loginfo("  第 %d 个人: 位置(%.2f, %.2f, %.2f) 速度(%.2f, %.2f, %.2f) 速率%.2f m/s", 
                             tracker.track_id,
                             position[0], position[1], position[2],
                             velocity[0], velocity[1], velocity[2],
                             speed)
    
    def publish_markers(self, clusters):
        """发布检测标记"""
        marker_array = MarkerArray()
        
        for i, cluster in enumerate(clusters):
            centroid = np.mean(cluster[:, :3], axis=0)
            
            marker = Marker()
            marker.header.frame_id = "livox_frame"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "detections"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = centroid[0]
            marker.pose.position.y = centroid[1]
            marker.pose.position.z = (np.min(cluster[:, 2]) + np.max(cluster[:, 2])) / 2
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = np.max(cluster[:, 2]) - np.min(cluster[:, 2])
            
            # 绿色表示检测
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.7
            
            marker.lifetime = rospy.Duration(0.2)
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
    
    def publish_tracks(self):
        """发布跟踪标记"""
        marker_array = MarkerArray()
        
        for tracker in self.trackers:
            if tracker.is_valid():
                position = tracker.kalman.get_position()
                velocity = tracker.kalman.get_velocity()
                
                # 位置标记
                marker = Marker()
                marker.header.frame_id = "livox_frame"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "tracks"
                marker.id = tracker.track_id
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                
                marker.pose.position.x = position[0]
                marker.pose.position.y = position[1]
                marker.pose.position.z = position[2]
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = 0.6
                marker.scale.y = 0.6
                marker.scale.z = 1.7
                
                # 红色表示跟踪
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.8
                
                marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(marker)
                
                # 速度箭头
                if np.linalg.norm(velocity) > 0.1:
                    arrow = Marker()
                    arrow.header.frame_id = "livox_frame"
                    arrow.header.stamp = rospy.Time.now()
                    arrow.ns = "velocity"
                    arrow.id = tracker.track_id
                    arrow.type = Marker.ARROW
                    arrow.action = Marker.ADD
                    
                    arrow.pose.position.x = position[0]
                    arrow.pose.position.y = position[1]
                    arrow.pose.position.z = position[2] + 0.5
                    
                    # 计算箭头方向
                    vel_norm = velocity / np.linalg.norm(velocity)
                    yaw = np.arctan2(vel_norm[1], vel_norm[0])
                    quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
                    arrow.pose.orientation.x = quat[0]
                    arrow.pose.orientation.y = quat[1]
                    arrow.pose.orientation.z = quat[2]
                    arrow.pose.orientation.w = quat[3]
                    
                    arrow.scale.x = min(2.0, np.linalg.norm(velocity))
                    arrow.scale.y = 0.1
                    arrow.scale.z = 0.1
                    
                    arrow.color.r = 0.0
                    arrow.color.g = 0.0
                    arrow.color.b = 1.0
                    arrow.color.a = 0.8
                    
                    arrow.lifetime = rospy.Duration(0.5)
                    marker_array.markers.append(arrow)
        
        self.tracks_pub.publish(marker_array)
    
    def publish_person_points(self, clusters, header):
        """发布检测到的人体点云"""
        if not clusters:
            return
            
        all_points = np.vstack(clusters)
        person_cloud = pc2.create_cloud_xyz32(header, all_points[:, :3])
        self.person_points_pub.publish(person_cloud)
    
    def lidar_callback(self, msg):
        """处理LiDAR点云数据"""
        start_time = time.time()
        
        if self.imu_data is None:
            rospy.logwarn_throttle(2.0, "等待IMU数据...")
            return
        
        # 将PointCloud2消息转换为numpy数组
        cloud_points = []
        for p in pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            cloud_points.append([p[0], p[1], p[2], p[3]])
        
        if len(cloud_points) == 0:
            return
            
        points = np.array(cloud_points)
        
        # 过滤点云
        filtered_points = self.filter_point_cloud(points)
        
        if len(filtered_points) < 10:
            return
        
        # 检测人体
        person_clusters = self.detect_persons(filtered_points)
        
        # 更新跟踪器
        self.update_trackers(person_clusters)
        
        # 记录和发布跟踪数据
        self.record_tracking_data()
        self.publish_tracking_data()
        
        # 发布可视化标记
        self.publish_markers(person_clusters)
        self.publish_tracks()
        
        # 发布人体点云
        self.publish_person_points(person_clusters, msg.header)
        
        # 输出检测结果和处理时间
        proc_time = time.time() - start_time
        valid_tracks = len([t for t in self.trackers if t.is_valid()])
        rospy.loginfo("检测: %d 个人体, 跟踪: %d 个目标. 处理时间: %.3f 秒", 
                     len(person_clusters), valid_tracks, proc_time)

if __name__ == '__main__':
    try:
        detector = PersonDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass