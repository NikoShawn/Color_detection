#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import message_filters # 用于同步图像话题

# --- 配置参数 ---
# 定义要识别的颜色范围 (HSV)
# 示例：识别红色 (你需要根据实际情况调整这些值)
# 红色有两个 HSV 范围，这里使用第一个范围（接近 0 度的红色）
LOWER_COLOR = np.array([0, 100, 100])   # HSV 下限
UPPER_COLOR = np.array([10, 255, 255])  # HSV 上限

# 最小轮廓面积，以过滤掉小的噪声
MIN_CONTOUR_AREA = 500

# 全局变量
bridge = CvBridge()
latest_average_depth = 0.0
object_detected_flag = False

def image_callback(color_msg, depth_msg):
    """
    当接收到同步的彩色和深度图像时调用此回调函数。
    """
    global latest_average_depth, object_detected_flag

    try:
        # 将 ROS Image 消息转换为 OpenCV 图像
        # color_image 通常是 BGR8 格式
        color_image = bridge.imgmsg_to_cv2(color_msg, "bgr8")
        # depth_image 通常是 16UC1 (毫米) 或 32FC1 (米)
        # realsense-ros align_depth:=true 通常输出 16UC1 格式的深度图，单位是毫米
        depth_image_raw = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

        # 始终显示原始视频
        cv2.imshow("Original Video", color_image)

        if depth_msg.encoding == '16UC1':
            # 将毫米转换为米以便后续处理，并转换为浮点数
            depth_image = depth_image_raw.astype(np.float32) / 1000.0
        elif depth_msg.encoding == '32FC1':
            depth_image = depth_image_raw # 已经是米了
        else:
            rospy.logwarn_throttle(1.0, f"Unsupported depth image encoding: {depth_msg.encoding}. Assuming 16UC1 or 32FC1.")
            # 根据实际情况处理或报错退出
            # 为简单起见，这里尝试按16UC1处理，如果不对，深度值会不准确
            depth_image = depth_image_raw.astype(np.float32) / 1000.0


    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return

    # 1. 颜色识别
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, LOWER_COLOR, UPPER_COLOR)

    # 形态学操作 (可选，但推荐)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 2. 查找轮廓
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_object_detected = False
    current_average_depth = 0.0

    if contours:
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]

        if valid_contours:
            # 创建一个合并的掩码，包含所有有效的轮廓区域
            combined_mask_for_depth = np.zeros_like(mask, dtype=np.uint8)
            cv2.drawContours(combined_mask_for_depth, valid_contours, -1, (255), thickness=cv2.FILLED)

            # 3. 提取并计算平均深度
            # 确保深度图像和掩码的尺寸一致
            if depth_image.shape[:2] == combined_mask_for_depth.shape[:2]:
                masked_depth_values = depth_image[combined_mask_for_depth > 0]
                valid_depth_values = masked_depth_values[np.isfinite(masked_depth_values) & (masked_depth_values > 0)] # 过滤 NaN, inf 和零值

                if valid_depth_values.size > 0:
                    current_average_depth = np.mean(valid_depth_values)
                    current_object_detected = True
            else:
                rospy.logwarn_throttle(1.0, "Depth image and mask dimensions do not match!")


            # 可选: 在彩色图像上绘制轮廓以便可视化
            display_image = color_image.copy()
            for contour in valid_contours:
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                cv2.drawContours(display_image, [contour], -1, (0, 255, 0), 2)
                cv2.circle(display_image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                if current_object_detected:
                    cv2.putText(display_image, f"Depth: {current_average_depth:.2f}m", (int(x)-50, int(y)-20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.imshow("Color Detection", display_image)

    # 如果没有有效轮廓，也显示原始图像或一个空白的提示
    if not current_object_detected and 'display_image' not in locals():
        cv2.imshow("Color Detection", color_image) # 或者显示 mask: cv2.imshow("Mask", mask)

    # 更新全局变量以供主循环打印
    latest_average_depth = current_average_depth
    object_detected_flag = current_object_detected

    cv2.waitKey(1) # 必须有 waitKey 才能让 imshow 正常工作

def main():
    global latest_average_depth, object_detected_flag

    rospy.init_node('color_depth_detector_node', anonymous=True)
    rospy.loginfo("Color Depth Detector Node Started")

    # --- ROS 主题名称 ---
    # 这些是 realsense-ros 包通常发布的主题。请根据你的实际情况调整。
    # 使用 align_depth:=true 时，深度图会对齐到彩色摄像头
    color_topic = "/camera/color/image_raw"
    depth_topic = "/camera/aligned_depth_to_color/image_raw" # 关键：使用对齐后的深度图

    # 创建订阅者
    # message_filters 用于同步来自不同主题的消息
    color_sub = message_filters.Subscriber(color_topic, Image)
    depth_sub = message_filters.Subscriber(depth_topic, Image)

    # ApproximateTimeSynchronizer 允许多个消息的时间戳有小的差异
    # queue_size 定义了消息队列的大小，slop 定义了消息之间允许的最大时间差（秒）
    ts = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], queue_size=10, slop=0.1)
    ts.registerCallback(image_callback)

    rate = rospy.Rate(10) # 10 Hz，控制打印频率

    # 创建 OpenCV 窗口
    cv2.namedWindow("Original Video", cv2.WINDOW_AUTOSIZE)  # 新增：原始视频窗口
    cv2.namedWindow("Color Detection", cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow("Mask", cv2.WINDOW_AUTOSIZE) # 如果需要单独看掩码

    try:
        while not rospy.is_shutdown():
            if object_detected_flag:
                rospy.loginfo(f"Detected color object - Average depth: {latest_average_depth:.2f} meters")
            else:
                rospy.loginfo("No specified color object detected or object too small.")
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down color_depth_detector_node.")
    finally:
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()