import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters

class DepthOfRed:
    def __init__(self):
        self.bridge = CvBridge()
        self.color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)

        # 使用 ApproximateTimeSynchronizer 来同步具有相似时间戳的消息
        # 调整 slop 参数（以秒为单位）以允许时间戳之间的一些差异
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1  # 0.1 秒的容差
        )
        self.ts.registerCallback(self.callback)
        rospy.loginfo("订阅了颜色和深度图像话题，并同步它们。")

    def callback(self, color_msg, depth_msg):
        rospy.loginfo("接收到同步的图像消息。")
        try:
            # 将 ROS Image 消息转换为 OpenCV 图像
            color_image = self.bridge.imgmsg_to_cv2(color_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough") # 通常是 16UC1 或 32FC1
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge 错误: {e}")
            return

        # 定义红色的 HSV 范围
        # 这些值可能需要根据您的具体照明条件和相机进行调整
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([5, 255, 255])
        lower_red2 = np.array([180, 70, 50])
        upper_red2 = np.array([180, 255, 255])

        # 将 BGR 图像转换为 HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # 创建红色区域的掩码
        mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 查找红色像素的坐标
        red_pixel_coordinates = np.argwhere(red_mask > 0) # 返回 (row, col) 形式的索引

        if red_pixel_coordinates.size > 0:
            rospy.loginfo(f"检测到 {len(red_pixel_coordinates)} 个红色像素。")
            valid_depths = []
            for coord in red_pixel_coordinates:
                row, col = coord
                # 获取对应像素的深度值
                # 深度图像的单位通常是毫米 (对于 16UC1) 或米 (对于 32FC1)
                # 请查阅您的相机文档以确认单位
                depth_value = depth_image[row, col]
                
                # 仅处理有效的深度值（例如，非零值，具体取决于您的深度传感器）
                if depth_value > 0: 
                    valid_depths.append(depth_value)
                    # rospy.loginfo(f"红色像素位于 ({col}, {row})，深度: {depth_value}")
                    # 在这里您可以对深度信息进行进一步处理
            
            if valid_depths:
                average_depth = np.mean(valid_depths)
                rospy.loginfo(f"检测到的红色区域的平均深度: {average_depth}")
            else:
                rospy.loginfo("检测到红色像素，但没有有效的深度值。")
        else:
            rospy.loginfo("未检测到红色像素。")

        # （可选）显示图像以进行调试
        # cv2.imshow("Color Image", color_image)
        # cv2.imshow("Depth Image", depth_image / np.max(depth_image)) # 归一化以便显示
        # cv2.imshow("Red Mask", red_mask)
        # cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('depth_of_red_detector', anonymous=True)
    dor = DepthOfRed()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("关闭节点。")
    # cv2.destroyAllWindows() # 如果使用了 cv2.imshow