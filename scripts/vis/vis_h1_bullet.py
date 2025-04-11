import pybullet as pb
import pybullet_data
import time
import joblib
import numpy as np

# Configuration class to store dataset path
class RetargetConfig:
    def __init__(self):
        # Path to the retargeted dataset pickle file
        self.SAVE_PATH = "data/h1/amass_partial_v3.pkl"

# Load the retargeted dataset
config = RetargetConfig()
data_dump = joblib.load(config.SAVE_PATH)

# Connect to PyBullet in GUI mode for visualization
pb.connect(pb.GUI)

# Set the search path for PyBullet data (e.g., plane.urdf)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a ground plane for reference
plane_id = pb.loadURDF("plane.urdf")

# Load the H1 robot URDF (assumed to be floating base since no world joint is defined)
robot_id = pb.loadURDF("resources/robots/h1/urdf/h1.urdf", useFixedBase=False)

# Adjust the camera for better viewing
pb.resetDebugVisualizerCamera(
    cameraDistance=2,
    cameraYaw=0,
    cameraPitch=-20,
    cameraTargetPosition=[0, 0, 1]
)

def visualize_sequence(sequence_data, robot_id, fps):
    """
    Visualize a single sequence of motion on the H1 robot.
    
    Args:
        sequence_data (dict): Dictionary containing 'root_trans_offset', 'root_rot', 'dof', and 'fps'.
        robot_id (int): PyBullet ID of the H1 robot.
        fps (float): Frames per second for playback.
    """
    # Extract data from the sequence
    root_trans = sequence_data['root_trans_offset']  # Shape: (N, 3)
    root_rot = sequence_data['root_rot']              # Shape: (N, 4)
    dof = sequence_data['dof']                        # Shape: (N, 19)
    N = root_trans.shape[0]                           # Number of frames
    dt = 1.0 / fps                                    # Time step per frame

    # Animate the sequence frame by frame
    for i in range(N):
        # Set the base position and orientation (convert numpy arrays to lists for PyBullet)
        pb.resetBasePositionAndOrientation(
            robot_id,
            posObj=root_trans[i].tolist(),
            ornObj=root_rot[i].tolist()
        )
        
        # Set the joint angles for the 19 actuated joints (indices 0 to 18)
        for j in range(19):
            pb.resetJointState(
                robot_id,
                jointIndex=j,
                targetValue=dof[i, j]
            )
        
        # Sleep to control playback speed
        time.sleep(dt)

# Visualize all sequences in the dataset
for key in data_dump:
    print(f"Visualizing sequence: {key}")
    sequence_data = data_dump[key]
    visualize_sequence(sequence_data, robot_id, sequence_data['fps'])

# Disconnect from PyBullet when done
pb.disconnect()