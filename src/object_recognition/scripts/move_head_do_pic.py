#!/usr/bin/env python3
import rospy
import cv2
import tf
import numpy as np
import open3d as o3d
import os
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Empty, Bool

# ---------------- Global Variables ----------------
k = 0
cv_image_rgb = None
gray_image = None
cloud_msg = None
bridge = CvBridge()
show_window = False  # Set to True to see images
base_path = "/home/pal/Desktop/DATA/picture_scene"
os.makedirs(base_path, exist_ok=True)

# State management
pub_done = None
acquisition_in_progress = False
acquisition_complete = False
acquisition_lock = threading.Lock()

# ---------------- Movement Configurations ----------------
torso_max = 0.3
head_fixed_angle = -0.9  # Fixed head_2_joint
head_1_start = -0.5
head_1_end = 0.5
head_1_delta = 0.5

# ---------------- Image Callbacks ----------------
def rgbCallback(msg):
    global cv_image_rgb
    try:
        cv_image_rgb = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if show_window:
            cv2.imshow("RGB Camera", cv_image_rgb)
            cv2.waitKey(1)
    except Exception as e:
        rospy.logwarn(f"Error processing RGB image: {e}")

def depthCallback(msg):
    global gray_image
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        # Replace NaN and infinite values
        cv_image = np.nan_to_num(cv_image, nan=0.0, posinf=0.0, neginf=0.0)
        if np.any(cv_image > 0):
            normalized = cv2.normalize(cv_image, None, 0, 1, cv2.NORM_MINMAX)
            gray_image = (normalized * 255).astype(np.uint8)
        else:
            rospy.logwarn("No valid depth data received")
            gray_image = np.zeros(cv_image.shape, dtype=np.uint8)
    except Exception as e:
        rospy.logwarn(f"Error processing depth image: {e}")
        gray_image = None

def pointsCallback(msg):
    global cloud_msg
    cloud_msg = msg

# ---------------- Helper Functions ----------------
def save_pointcloud(msg, filename="cloud.pcd"):
    points = []
    for p in pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z")):
        points.append([p[0], p[1], p[2]])
    if points:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        o3d.io.write_point_cloud(filename, pcd)

def save_data(mode='w'):
    global k
    if cv_image_rgb is None or gray_image is None or cloud_msg is None:
        rospy.logwarn("Data not available for saving!")
        return False

    try:
        filename_pointcloud = os.path.join(base_path, f"depth_registered_points{k}.pcd")
        save_pointcloud(cloud_msg, filename_pointcloud)

        filename_rgb = os.path.join(base_path, f"rgb_image_color_{k}.png")
        cv2.imwrite(filename_rgb, cv_image_rgb)

        listener = tf.TransformListener()
        listener.waitForTransform("base_footprint", "xtion_rgb_optical_frame",
                                rospy.Time(0), rospy.Duration(3.0))
        (trans, rot) = listener.lookupTransform("base_footprint",
                                              "xtion_rgb_optical_frame",
                                              rospy.Time(0))

        # Salva anche le informazioni della camera
        try:
            camera_info = rospy.wait_for_message("/xtion/rgb/camera_info", CameraInfo, timeout=1.0)
            camera_info_file = os.path.join(base_path, "camera_info.txt")
            with open(camera_info_file, mode) as f:
                f.write(f"{camera_info.width},{camera_info.height}\n")
                K = np.array(camera_info.K).reshape(3, 3)
                for row in K:
                    f.write(",".join(map(str, row)) + "\n")
            rospy.loginfo(f"Saved camera info to {camera_info_file}")
        except rospy.ROSException as e:
            rospy.logwarn(f"Could not get camera info: {e}")

        with open(os.path.join(base_path, "transform_info.txt"), mode) as f:
            f.write(f"{trans[0]}, {trans[1]}, {trans[2]}, "
                   f"{rot[0]}, {rot[1]}, {rot[2]}, {rot[3]}\n")

        rospy.loginfo(f"Saved set {k}")
        k += 1
        return True
    except Exception as e:
        rospy.logwarn(f"Error saving data: {e}")
        return False

def publish_trajectory(pub, joint_names, positions, duration=2.0):
    traj = JointTrajectory()
    traj.joint_names = joint_names

    point = JointTrajectoryPoint()
    point.positions = positions
    point.time_from_start = rospy.Duration(duration)
    traj.points.append(point)

    pub.publish(traj)
    rospy.loginfo(f"Published command to {pub.resolved_name}: {positions}")
    rospy.sleep(duration + 0.5)

# ---------------- Main Acquisition ----------------
def main_acquisition():
    global acquisition_in_progress, acquisition_complete, pub_done
    
    try:
        rospy.loginfo("Starting head sweep and image capture")

        transform_file = os.path.join(base_path, "transform_info.txt")
        if os.path.exists(transform_file):
            os.remove(transform_file)

        rospy.sleep(1.0)

        head_pub = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=10)
        rospy.sleep(2.0)  # Give time for publisher to register

        if not save_data(mode='a'):
            rospy.logerr("Failed to save initial data")
            raise Exception("Failed to save initial data")

        h2 = head_fixed_angle
        h1 = head_1_start
        while h1 <= head_1_end and not rospy.is_shutdown():
            publish_trajectory(head_pub, ["head_1_joint", "head_2_joint"], [h1, h2], duration=1.0)
            rospy.sleep(5.0)
            if not save_data(mode='a'):
                rospy.logwarn("Failed to save data, continuing...")
            h1 += head_1_delta

        rospy.loginfo("Acquisition completed successfully!")
        pub_done.publish(True)
        
    except Exception as e:
        rospy.logerr(f"Error during acquisition: {str(e)}")
        pub_done.publish(False)
        raise
    finally:
        with acquisition_lock:
            acquisition_in_progress = False
            acquisition_complete = True

# ---------------- Topic Handler ----------------
def handle_take_photos(msg):
    global acquisition_in_progress, acquisition_complete
    
    with acquisition_lock:
        if acquisition_in_progress:
            rospy.logwarn("Acquisition already in progress")
            return
            
        if acquisition_complete:
            rospy.loginfo("Restarting acquisition...")
            acquisition_complete = False  # Reset for new acquisition
            
        acquisition_in_progress = True
        rospy.loginfo("Received start acquisition command...")
        
        try:
            t = threading.Thread(target=main_acquisition)
            t.daemon = True
            t.start()
        except Exception as e:
            acquisition_in_progress = False
            rospy.logerr(f"Error starting acquisition: {str(e)}")

# ---------------- Main ----------------
def main():
    global pub_done
    
    rospy.init_node("tiago_rgbd_capture_topic", anonymous=True)
    
    # Initialize subscribers
    rospy.Subscriber("/xtion/rgb/image_raw", Image, rgbCallback)
    rospy.Subscriber("/xtion/depth_registered/image_raw", Image, depthCallback)
    rospy.Subscriber("/xtion/depth_registered/points", PointCloud2, pointsCallback)
    rospy.Subscriber("/start_acquisition", Empty, handle_take_photos)

    # Initialize publisher
    pub_done = rospy.Publisher('/take_photos_done', Bool, queue_size=1, latch=True)
    
    rospy.loginfo("Node ready. Publish to /start_acquisition to begin.")
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted by user")
    except Exception as e:
        rospy.logerr(f"Error in main node: {str(e)}")
    finally:
        rospy.loginfo("Shutting down node...")

if __name__ == "__main__":
    main()