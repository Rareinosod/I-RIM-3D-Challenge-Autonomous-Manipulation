#!/usr/bin/env python3
import rospy, json, sys, copy, numpy as np, tf, tf2_ros, moveit_commander
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from std_msgs.msg import String, Float32, Bool

class R2HExecutor:
    def __init__(self):
        rospy.init_node("r2h_executor_node", anonymous=True)

        # MoveIt init
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm = moveit_commander.MoveGroupCommander("arm_torso")
        self.gripper = moveit_commander.MoveGroupCommander("gripper")

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Parametri
        self.PREGRASP_HEIGHT = 0.20
        self.GRIPPER_OFFSET = 0.22
        self.GRIPPER_OPEN = [0.04, 0.04]
        self.TABLE_Z = 0.5
        self.SAFETY_MARGIN = 0.0

        # Carico oggetti
        json_path = "/home/pal/Desktop/DATA/picture_scene/object_poses.json"
        with open(json_path, "r") as f:
            self.objects_data = json.load(f)

        # Variabili runtime
        self.hand_pose_msg = None
        self.executing = False  # Flag per bloccare aggiornamenti hand_box durante esecuzione

        # Publisher e Subscriber
        self.place_pose_pub = rospy.Publisher("/place_goal_pose", PoseStamped, queue_size=1)
        self.width_pub = rospy.Publisher("/previous_width", Float32, queue_size=1)
        self.r2h_done_pub = rospy.Publisher("/R2H_done", Bool, queue_size=1, latch=True)
        rospy.Subscriber("/object2detect", String, self.object_callback)
        rospy.Subscriber("/hand_pose", PoseStamped, self.hand_pose_callback)

        # Aggiungo table_box 
        self.add_collision_boxes()

    # ------------------ UTILITY ------------------
    def normalize(self, v):
        n = np.linalg.norm(v)
        return v if n < 1e-9 else v / n

    def reach_pose(self, group, pose):
        group.set_pose_target(pose)
        group.go(wait=True)
        group.stop()
        group.clear_pose_targets()

    def reach_waypoints(self, group, waypoints):
        plan, _ = group.compute_cartesian_path(waypoints, 0.01, 0.0)
        if plan:
            group.execute(plan, wait=True)
        group.stop()
        group.clear_pose_targets()

    def open_grippers(self):
        self.gripper.go(self.GRIPPER_OPEN, wait=True)
        self.gripper.stop()

    def close_grippers(self, width):
        self.gripper.go([width/2, width/2], wait=True)
        self.gripper.stop()

    def compute_topdown_quaternion_x_down_align_y(self, pose_obj):
        q_obj = [pose_obj.orientation.x, pose_obj.orientation.y, pose_obj.orientation.z, pose_obj.orientation.w]
        R_obj = tf.transformations.quaternion_matrix(q_obj)[:3,:3]
        z_obj = R_obj[:,2]
        z_proj = np.array([z_obj[0], z_obj[1], 0.0])
        if np.linalg.norm(z_proj) < 1e-4:
            z_proj = np.array([0.0,1.0,0.0])
        z_proj = self.normalize(z_proj)
        x_des = np.array([0.0,0.0,-1.0])
        y_des = self.normalize(z_proj)
        z_des = self.normalize(np.cross(x_des, y_des))
        R = np.eye(4)
        R[0:3,0] = x_des
        R[0:3,1] = y_des
        R[0:3,2] = z_des
        quat = tf.transformations.quaternion_from_matrix(R)
        return quat

    def compute_place_quaternion_y_up_z_forward_x_left(self):
        x_des = np.array([-1.0,0.0,0.0])
        y_des = np.array([0.0,1.0,0.0])
        z_des = np.array([0.0,0.0,1.0])
        R = np.eye(4)
        R[0:3,0] = x_des
        R[0:3,1] = y_des
        R[0:3,2] = z_des
        quat = tf.transformations.quaternion_from_matrix(R)
        return quat

    # ------------------ COLLISION BOX ------------------
    def add_collision_boxes(self):
        existing_objects = self.scene.get_known_object_names()
        if "table_box" in existing_objects:
            rospy.loginfo("table_box già presente, non riaggiungo.")
            return
        # Table box
        z_values = [obj["position"]["z"] for obj in self.objects_data.values() if "position" in obj]
        min_z = min(z_values)
        table_pose = PoseStamped()
        table_pose.header.frame_id = "base_footprint"
        table_pose.pose.position.x = min_z + 0.3
        table_pose.pose.position.y = 0
        table_pose.pose.position.z = min_z / 2
        table_pose.pose.orientation.w = 1.0
        table_size = (2*min_z, 2*min_z, min_z-0.01)
        self.scene.add_box("table_box", table_pose, size=table_size)

    # ------------------ CALLBACK ------------------
    def object_callback(self, msg):
        obj_name = msg.data.strip()
        
        # Ignora messaggi per H2R (Right/Left)
        if obj_name in ["Right", "Left"]:
            return
        
        if obj_name not in self.objects_data:
            rospy.logwarn(f"Oggetto '{obj_name}' non trovato nel JSON!")
            return
        self.grab_and_place(self.arm, self.gripper, self.objects_data[obj_name])

    def hand_pose_callback(self, msg):
        self.hand_pose_msg = copy.deepcopy(msg)
        hand_box_size = (0.20, 0.10, 0.05)
        hand_box_pose = PoseStamped()
        hand_box_pose.header.frame_id = "base_footprint"
        hand_box_pose.pose = msg.pose
        self.scene.add_box("hand_box", hand_box_pose, size=hand_box_size)

    # ------------------ PICK & PLACE ------------------
    def grab_and_place(self, group, gripper, obj_info):
        # Blocca aggiornamenti hand_box durante l'esecuzione
        self.executing = True
        
        pose_goal = Pose()
        pose_goal.position.x = obj_info["position"]["x"]
        pose_goal.position.y = obj_info["position"]["y"]
        pose_goal.position.z = obj_info["position"]["z"]
        pose_goal.orientation.x = obj_info["orientation"]["x"]
        pose_goal.orientation.y = obj_info["orientation"]["y"]
        pose_goal.orientation.z = obj_info["orientation"]["z"]
        pose_goal.orientation.w = obj_info["orientation"]["w"]

        width = obj_info.get("width", 0.04)  # width dal JSON
        self.width_pub.publish(Float32(data=width))

        quat_pick = self.compute_topdown_quaternion_x_down_align_y(pose_goal)
        q_pick = Quaternion(*quat_pick)

        pre_grasp = copy.deepcopy(pose_goal)
        pre_grasp.position.z = max(pose_goal.position.z + self.PREGRASP_HEIGHT + self.GRIPPER_OFFSET,
                                  self.TABLE_Z + self.PREGRASP_HEIGHT + 0.0) + self.GRIPPER_OFFSET
        pre_grasp.orientation = q_pick

        grasp_pose = copy.deepcopy(pose_goal)
        grasp_pose.position.z = max(pose_goal.position.z + self.GRIPPER_OFFSET,
                                    self.TABLE_Z + self.GRIPPER_OFFSET + 0.0)
        grasp_pose.orientation = q_pick

        # Apertura gripper sempre 0.04
        self.open_grippers()
        rospy.loginfo("→ Pre-grasp")
        self.reach_pose(group, pre_grasp)
        rospy.loginfo("→ Discesa sul punto di grasp")
        self.reach_waypoints(group, [grasp_pose])

        # Chiusura gripper width/2
        rospy.loginfo("→ Chiusura gripper")
        self.close_grippers(width)

        # Sollevamento
        rospy.loginfo("→ Sollevamento")
        self.reach_waypoints(group, [pre_grasp])
        rospy.sleep(0.5)

        # Attesa hand_pose
        rospy.loginfo("→ Attendo hand_pose per place...")
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and self.hand_pose_msg is None:
            rate.sleep()

        place_pose = copy.deepcopy(self.hand_pose_msg)
        place_pose.pose.position.z += 0.1
        place_pose.pose.position.x -= 0.22
        place_pose.pose.orientation = Quaternion(*self.compute_place_quaternion_y_up_z_forward_x_left())

        place_pose.header.stamp = rospy.Time.now()
        place_pose.header.frame_id = "base_footprint"
        self.place_pose_pub.publish(place_pose)
        rospy.loginfo("→ Movimento verso hand_pose + offset")
        self.reach_pose(group, place_pose.pose)

        # Apertura gripper sempre 0.04
        rospy.loginfo("→ Apertura gripper")
        self.open_grippers()

        # ------------------ MOVE RIGHT + HOME ------------------
        rospy.loginfo("→ Spostamento 0.20 m verso destra")
        current_pose = group.get_current_pose().pose
        new_pose = copy.deepcopy(current_pose)
        new_pose.position.y -= 0.20
        self.reach_pose(group, new_pose)
        rospy.sleep(0.5)

        rospy.loginfo("→ Torna in home position")
        home_joint_positions = [0.35, 0.07, 0.1, -3.1, 1.36, 2.05, 0.01, -0.05]
        group.go(home_joint_positions, wait=True)
        group.stop()
        rospy.loginfo("✅ Home position raggiunta")
        
        # Riabilita aggiornamenti hand_box
        self.executing = False
        
        rospy.sleep(0.5)
        self.r2h_done_pub.publish(True)

        self.scene.remove_world_object("hand_box")
        self.scene.remove_world_object("table_box")

        rospy.loginfo("Shutting down MoveIt cleanly...")
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
        rospy.loginfo("✅ Segnale R2H_done pubblicato")

# ------------------ MAIN ------------------
if __name__ == "__main__":
    executor = R2HExecutor()
    rospy.loginfo("Nodo R2H pronto. Pubblica su /object2detect per iniziare.")
    rospy.spin()
