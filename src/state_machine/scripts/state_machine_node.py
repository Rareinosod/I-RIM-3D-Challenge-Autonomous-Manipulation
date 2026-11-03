#!/usr/bin/env python3
import rospy
import subprocess
import time
import signal
import sys
from std_msgs.msg import String, Bool, Empty
from std_srvs.srv import Trigger, TriggerRequest
import os

class StateMachine:
    def __init__(self):
        rospy.init_node('state_machine')

        # Publishers
        self.gesture_enable_pub = rospy.Publisher('/gesture_enable', Bool, queue_size=1, latch=True)
        self.start_acquisition_pub = rospy.Publisher('/start_acquisition', Empty, queue_size=1, latch=True)

        # Subscribers
        rospy.Subscriber('/object2detect', String, self.object_callback)
        rospy.Subscriber('/R2H_done', Bool, self.r2h_done_callback)
        rospy.Subscriber('/H2R_done', Bool, self.h2r_done_callback)
        rospy.Subscriber('/take_photos_done', Bool, self.take_photos_done_callback, queue_size=1)
        rospy.Subscriber('/object_localization_done', Bool, self.object_localization_done_callback)

        #self.current_state = 'INIT' #rimettere
        self.current_state = 'WAIT_FOR_GESTURE' # togliere
        self.object_to_detect = None
        self.r2h_done = False
        self.h2r_done = False
        self.take_photos_done = False
        self.object_localization_done = False

        self.node_processes = {}

        # Node commands
        self.commands = {
            'move_head': 'rosrun object_recognition move_head_do_pic.py',
            'PCobj': 'rosrun object_recognition PC_objDetect_color.py',
            'gesture': 'rosrun gesture_recognition gesture_recogn_sim_cameraPC.py',  # Usa webcam PC per test
            'R2H': 'rosrun pick_and_place R2H.py',
            'H2R': 'rosrun pick_and_place H2R.py'
        }

    def object_callback(self, msg):
        if self.current_state in ['WAIT_FOR_GESTURE']:
            self.object_to_detect = msg.data
            rospy.loginfo(f"Received object: {self.object_to_detect}. Transitioning state.")

    def r2h_done_callback(self, msg):
        if msg.data:
            self.r2h_done = True
            rospy.loginfo("Received R2H completion signal")

    def h2r_done_callback(self, msg):
        if msg.data:
            self.h2r_done = True
            rospy.loginfo("Received H2R completion signal")

    def take_photos_done_callback(self, msg):
        if msg.data:
            self.take_photos_done = True
            rospy.loginfo("Received take_photos_done signal")
        else:
            rospy.logwarn("Acquisizione fallita, ricevuto take_photos_done=False")

    def object_localization_done_callback(self, msg):
        if msg.data:
            self.object_localization_done = True
            rospy.loginfo("Object localization completed")

    def start_node(self, name):
        if name in self.node_processes and self.node_processes[name].poll() is None:
            rospy.loginfo(f"Node {name} is already running.")
            return
        try:
            rospy.loginfo(f"Starting node: {name}")
            process = subprocess.Popen(self.commands[name], shell=True)
            self.node_processes[name] = process
            # Rimuovi il time.sleep fisso, verrà gestito manualmente dove necessario
        except Exception as e:
            rospy.logerr(f"Failed to start node {name}: {e}")

    def kill_node(self, name):
        if name in self.node_processes:
            try:
                if self.node_processes[name].poll() is None:
                    rospy.loginfo(f"Terminating node: {name}")
                    self.node_processes[name].terminate()
                    try:
                        self.node_processes[name].wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        rospy.logwarn(f"Node {name} didn't terminate gracefully, forcing...")
                        self.node_processes[name].kill()
                        self.node_processes[name].wait()
            except Exception as e:
                rospy.logerr(f"Error killing node {name}: {e}")
            finally:
                del self.node_processes[name]

    def cleanup(self):
        rospy.loginfo("Cleaning up resources...")
        # Disable gesture recognition if active
        self.gesture_enable_pub.publish(False)
        
        # Terminate all started nodes
        for name in list(self.node_processes.keys()):
            self.kill_node(name)

    def run(self):
        rate = rospy.Rate(1)  # 1 Hz
        try:
            while not rospy.is_shutdown():
                rospy.loginfo(f"Current state: {self.current_state}")

                """if self.current_state == 'INIT':
                    self.start_node('move_head')
                    rospy.loginfo("Node move_head started, waiting for it to be ready...")
                    rospy.sleep(5.0)  # Give more time for the node to initialize

                    rospy.loginfo("Publishing start_acquisition...")
                    self.start_acquisition_pub.publish(Empty())

                    self.take_photos_done = False
                    timeout = rospy.Time.now() + rospy.Duration(120)  # 2 minutes timeout

                    while not self.take_photos_done and not rospy.is_shutdown():
                        if rospy.Time.now() > timeout:
                            rospy.logerr("Timeout waiting for take_photos_done")
                            break
                        rospy.loginfo("Waiting for take_photos_done...")
                        rate.sleep()

                    if self.take_photos_done:
                        rospy.loginfo("Photo acquisition completed successfully")
                        # Kill move_head node after completion
                        self.kill_node('move_head')
                        self.current_state = 'DETECT_OBJECT'
                    else:
                        rospy.logerr("Failed to complete photo acquisition, retrying...")
                        rospy.sleep(1.0)

                elif self.current_state == 'DETECT_OBJECT':
                    self.start_node('PCobj')
                    rospy.loginfo("Node PCobj started, waiting for service...")
                    
                    # Aumenta il tempo di attesa per permettere al nodo di inizializzarsi completamente
                    rospy.sleep(5.0)  # Prima erano solo time.sleep(1) dopo start_node
                    
                    try:
                        rospy.wait_for_service('/run_object_localization', timeout=15.0)  # Aumenta anche il timeout
                        rospy.loginfo("Service /run_object_localization is available")
                        
                        run_localization = rospy.ServiceProxy('/run_object_localization', Trigger)
                        response = run_localization()
                        rospy.loginfo(f"Service response: {response.message}")
                        
                        if response.success:
                            self.object_localization_done = False
                            timeout = rospy.Time.now() + rospy.Duration(60)
                            
                            while not self.object_localization_done and not rospy.is_shutdown():
                                if rospy.Time.now() > timeout:
                                    rospy.logerr("Timeout waiting for object localization")
                                    break
                                rospy.loginfo("Waiting for object localization to complete...")
                                rate.sleep()
                            
                            if self.object_localization_done:
                                rospy.loginfo("Object localization completed successfully")
                                # Kill PCobj node after completion (it also self-shutdowns)
                                rospy.sleep(1.0)  # Give time for node to complete shutdown
                                if 'PCobj' in self.node_processes:
                                    self.kill_node('PCobj')
                                # Start gesture recognition node once for the entire cycle
                                self.start_node('gesture')
                                rospy.sleep(2.0)  # Give time for gesture node to initialize
                                self.current_state = 'WAIT_FOR_GESTURE'
                            else:
                                rospy.logerr("Failed to complete object localization")
                        else:
                            rospy.logerr("Object localization service call failed")
                            
                    except (rospy.ServiceException, rospy.ROSException) as e:
                        rospy.logerr(f"Error in DETECT_OBJECT state: {e}")
                        rospy.sleep(1.0)"""
                
                if self.current_state == 'WAIT_FOR_GESTURE': # elif
                    self.start_node('gesture')
                    # Enable gesture recognition
                    rospy.loginfo("Enabling gesture recognition...")
                    self.gesture_enable_pub.publish(True)
                    self.object_to_detect = None
                    
                    while self.object_to_detect is None and not rospy.is_shutdown():
                        rospy.loginfo("Waiting for gesture...")
                        rate.sleep()
                    
                    if not rospy.is_shutdown():
                        rospy.loginfo(f"Gesture detected: {self.object_to_detect}")
                        # Decide immediately based on gesture type
                        if self.object_to_detect in ["Left", "Right"]:
                            rospy.loginfo("Detected Left/Right - executing H2R")
                            # Disable gesture and go directly to H2R
                            self.gesture_enable_pub.publish(False)
                            rospy.sleep(0.5)
                            self.current_state = 'EXECUTE_H2R'
                        else:
                            rospy.loginfo(f"Detected object: {self.object_to_detect} - executing R2H")
                            # Disable gesture and go directly to R2H
                            self.gesture_enable_pub.publish(False)
                            rospy.sleep(0.5)
                            self.current_state = 'EXECUTE_R2H'

                elif self.current_state == 'EXECUTE_R2H':
                    # Reset flag BEFORE starting node
                    self.r2h_done = False
                    self.start_node('R2H')
                    rospy.loginfo("R2H node started, waiting for it to subscribe to /object2detect...")
                    rospy.sleep(3.0)  # Give time for R2H to register subscribers and receive latched message
                    rospy.loginfo(f"R2H should have received object: {self.object_to_detect}")
                    
                    while not self.r2h_done and not rospy.is_shutdown():
                        rospy.loginfo("Executing R2H...")
                        rate.sleep()
                    
                    if not rospy.is_shutdown():
                        rospy.loginfo("R2H completed, killing node...")
                        self.kill_node('R2H')
                        self.current_state = 'WAIT_FOR_GESTURE'

                elif self.current_state == 'EXECUTE_H2R':
                    # Reset flag BEFORE starting node
                    self.h2r_done = False
                    self.start_node('H2R')
                    rospy.loginfo("H2R node started, waiting for it to subscribe to /object2detect...")
                    rospy.sleep(3.0)  # Give time for H2R to register subscribers and receive latched message
                    rospy.loginfo(f"H2R should have received side: {self.object_to_detect}")
                    
                    while not self.h2r_done and not rospy.is_shutdown():
                        rospy.loginfo("Executing H2R...")
                        rate.sleep()
                    
                    if not rospy.is_shutdown():
                        rospy.loginfo("H2R completed, killing node...")
                        self.kill_node('H2R')
                        self.current_state = 'WAIT_FOR_GESTURE'

                rate.sleep()

        except Exception as e:
            rospy.logerr(f"Error in state machine: {str(e)}")
        finally:
            self.cleanup()

def signal_handler(sig, frame):
    rospy.loginfo("Shutdown signal received, cleaning up...")
    rospy.signal_shutdown("Shutdown signal received")
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    try:
        sm = StateMachine()
        sm.run()
    except rospy.ROSInterruptException:
        pass