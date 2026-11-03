#!/usr/bin/env python3
import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
import tf
import sensor_msgs.point_cloud2 as pc2
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# =================== GLOBAL VARS =====================
bridge = CvBridge()
latest_rgb = None
latest_points = None
gesture_enabled = True  # default True
selection_locked = False  # reset when gesture re-enabled

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

VALID_GESTURES = {
    "screwdriver_green", "screwdriver_red", "screwdriver_black",
    "hammer", "wrench", "pliers", "Left", "Right"
}

# =================== CALLBACKS =======================
def rgb_callback(msg):
    global latest_rgb
    latest_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

def points_callback(msg):
    global latest_points
    latest_points = msg

def enable_callback(msg):
    global gesture_enabled, selection_locked
    gesture_enabled = msg.data
    if gesture_enabled:
        # Reset selection_locked quando gesture viene riabilitato
        selection_locked = False
    rospy.loginfo(f"Gesture recognition {'ENABLED' if gesture_enabled else 'DISABLED'}")

# =================== HELPER FUNCTIONS =======================
def fingers_up(landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    fingers = [False] * 5
    if landmarks[tips_ids[0]][0] < landmarks[2][0]:
        fingers[0] = True
    for i in range(1, 5):
        if landmarks[tips_ids[i]][1] < landmarks[pip_ids[i]][1]:
            fingers[i] = True
    return fingers

def classify_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers
    if thumb and not any([index, middle, ring, pinky]): return "screwdriver_green"
    if thumb and index and not any([middle, ring, pinky]): return "screwdriver_red"
    if thumb and index and middle and not any([ring, pinky]): return "screwdriver_black"
    if all(fingers): return "hammer"
    if index and pinky and not any([thumb, middle, ring]): return "wrench"
    if thumb and pinky and not any([index, middle, ring]): return "pliers"
    if not any(fingers): return "h2r"
    return None

def get_xyz_from_pointcloud(cloud_msg, u, v, width):
    if cloud_msg is None: return None
    index = v * width + u
    gen = pc2.read_points(cloud_msg, field_names=("x","y","z"), skip_nans=False)
    for i, p in enumerate(gen):
        if i == index:
            if np.isfinite(p[0]) and np.isfinite(p[1]) and np.isfinite(p[2]):
                return np.array([p[0], p[1], p[2]])
            else:
                return None
    return None

# =================== MAIN NODE =======================
def main():
    global gesture_enabled, latest_rgb, latest_points, selection_locked
    rospy.init_node("gesture_from_xtion", anonymous=False)

    # Subscriber
    rospy.Subscriber("/xtion/rgb/image_raw", Image, rgb_callback)
    rospy.Subscriber("/xtion/depth_registered/points", PointCloud2, points_callback)
    rospy.Subscriber("/gesture_enable", Bool, enable_callback)

    gesture_pub = rospy.Publisher("/object2detect", String, queue_size=10, latch=True)  # LATCH per mantenere ultimo messaggio
    hand_pose_pub = rospy.Publisher("/hand_pose", PoseStamped, queue_size=10)
    head_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=1)

    # Metti testa dritta prima di partire
    traj_msg = JointTrajectory()
    traj_msg.joint_names = ["head_1_joint", "head_2_joint"]
    point = JointTrajectoryPoint()
    point.positions = [0.0, 0.0]
    point.time_from_start = rospy.Duration(1.0)
    traj_msg.points.append(point)
    head_pub.publish(traj_msg)
    rospy.loginfo("Head set to straight position.")

    listener = tf.TransformListener()
    rate = rospy.Rate(10)
    try:
        listener.waitForTransform("base_footprint", "xtion_rgb_optical_frame", rospy.Time(0), rospy.Duration(5.0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logwarn("Trasformazione TF non trovata.")
        return

    last_gesture = None
    countdown = 0
    frame_threshold = 20
    confirmed_gesture = None

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) as hands:
        while not rospy.is_shutdown():
            if latest_rgb is None:
                rate.sleep()
                continue

            frame = latest_rgb.copy()
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # --- Pubblica hand pose anche se gesture disabilitata, mantiene ultimo valore ---
                    wrist_x = int(hand_landmarks.landmark[0].x * w)
                    wrist_y = int(hand_landmarks.landmark[0].y * h)
                    wrist_xyz = get_xyz_from_pointcloud(latest_points, wrist_x, wrist_y, w) if latest_points else None
                    if wrist_xyz is not None:
                        try:
                            (trans, rot) = listener.lookupTransform("base_footprint", "xtion_rgb_optical_frame", rospy.Time(0))
                            R = tf.transformations.quaternion_matrix(rot)[:3, :3]
                            wrist_in_base = np.dot(R, wrist_xyz) + np.array(trans)
                            pose_msg = PoseStamped()
                            pose_msg.header.stamp = rospy.Time.now()
                            pose_msg.header.frame_id = "base_footprint"
                            pose_msg.pose.position.x = wrist_in_base[0]
                            pose_msg.pose.position.y = wrist_in_base[1]
                            pose_msg.pose.position.z = wrist_in_base[2]
                            pose_msg.pose.orientation.w = 1.0
                            # pubblica sempre l'ultimo valore, anche se gesture_disabled
                            hand_pose_pub.publish(pose_msg)
                        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                            pass

                    # --- Gesture recognition solo se abilitata ---
                    if gesture_enabled:
                        lm = [(l.x, l.y) for l in hand_landmarks.landmark]
                        fingers = fingers_up(lm)
                        gesture = classify_gesture(fingers)
                        if gesture == "h2r":
                            gesture = handedness.classification[0].label
                        if gesture in VALID_GESTURES and not selection_locked:
                            if gesture == last_gesture:
                                countdown += 1
                            else:
                                countdown = 1
                                last_gesture = gesture
                            if countdown >= frame_threshold:
                                confirmed_gesture = gesture
                                selection_locked = True
                                gesture_pub.publish(confirmed_gesture)
                        else:
                            countdown = 0
                            last_gesture = None

            rate.sleep()

if __name__ == "__main__":
    main()
