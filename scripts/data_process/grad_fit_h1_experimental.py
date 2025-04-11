import glob
import pinocchio as pin
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from phc.smpllib.smpl_parser import SMPL_Parser, SMPL_BONE_ORDER_NAMES
import joblib
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch
from torch.autograd import Variable
from tqdm import tqdm
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

# Configuration class
class RetargetConfig:
    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.AMASS_ROOT = "/home/human2humanoid/data/AMASS/AMASS_Partial"
        self.SAVE_PATH = "data/h1/amass_partial_v3.pkl"
        self.URDF_PATH = "resources/robots/h1/urdf/h1.urdf"
        self.SMPL_MODEL_PATH = "data/smpl"
        self.SHAPE_PATH = "data/h1/shape_optimized_v3.pkl"
        self.FPS = 30
        self.DT = 1 / self.FPS
        self.MAX_ITERATIONS = 500
        self.LR = 100  # Reasonable value for Adadelta
        self.LOSS_WEIGHTS = {
            'pos': 1.0,         # Position loss weight
            'torque_penalty': 0.1, # Torque penalty weight
            'joint_limit_penalty': 0.1,  # Joint limit penalty weight
        }
        # Torque limits from H1 documentation
        self.H1_TORQUE_LIMITS = [
            220,  # right_hip_yaw_joint
            220,  # right_hip_roll_joint
            220,  # right_hip_pitch_joint
            360,  # right_knee_joint
            59,   # right_ankle_joint
            220,  # left_hip_yaw_joint
            220,  # left_hip_roll_joint
            220,  # left_hip_pitch_joint
            360,  # left_knee_joint
            59,   # left_ankle_joint
            220,  # torso_joint
            75,   # right_shoulder_pitch_joint
            75,   # right_shoulder_roll_joint
            75,   # right_shoulder_yaw_joint
            75,   # right_elbow_joint
            75,   # left_shoulder_pitch_joint
            75,   # left_shoulder_roll_joint
            75,   # left_shoulder_yaw_joint
            75    # left_elbow_joint
        ]
        self.EARLY_STOP_PATIENCE = 20    # Iterations to wait for loss improvement
        self.EARLY_STOP_THRESHOLD = 0.001 # Minimum loss improvement
        self.NUM_WORKERS = 3  # Added for multiprocessing

def load_amass_data(data_path):
    """Load AMASS motion data from an NPZ file."""
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if 'mocap_framerate' in entry_data:
        framerate = entry_data['mocap_framerate']
    elif 'mocap_frame_rate' in entry_data: # SOMA AND GRAB has this 
        framerate = entry_data['mocap_frame_rate']
    else:
        return {"flag": "missing_framerate"}

    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate
    }

def compute_torques(dof_pos_new, robot, data, device):
    """Compute gravity compensation torques using Pinocchio."""
    N = dof_pos_new.shape[1]
    torques = []
    for t in range(N):
        q = dof_pos_new[0, t, :, 0].cpu().detach().numpy()
        dq = np.zeros_like(q)  # Zero velocity
        ddq = np.zeros_like(q)  # Zero acceleration
        tau = pin.rnea(robot, data, q, dq, ddq)
        torques.append(tau)
    torques = np.stack(torques, axis=0)
    return torch.from_numpy(torques).to(device)

class Retargeter:
    def __init__(self, config: RetargetConfig):
        self.config = config
        self.device = config.DEVICE
        self._setup_models()
        self._setup_h1_limits()
        self._define_joint_mappings()

    def _setup_models(self):
        """Initialize SMPL parser, H1 FK, and Pinocchio models."""
        self.smpl_parser_n = SMPL_Parser(model_path=self.config.SMPL_MODEL_PATH, gender="neutral").to(self.device)
        self.h1_fk = Humanoid_Batch(device=self.device)
        self.robot = pin.buildModelFromUrdf(self.config.URDF_PATH)
        self.data = self.robot.createData()

    def _setup_h1_limits(self):
        """Set up torque limits."""
        self.h1_torque_limits = torch.tensor(self.config.H1_TORQUE_LIMITS, dtype=torch.float, device=self.device)

    def _define_joint_mappings(self):
        """Define H1 joint rotation axes and joint mappings."""
        self.h1_rotation_axis = torch.tensor([[
            [0, 0, 1],  # left_hip_yaw
            [1, 0, 0],  # left_hip_roll
            [0, 1, 0],  # left_hip_pitch
            [0, 1, 0],  # left_knee
            [0, 1, 0],  # left_ankle
            [0, 0, 1],  # right_hip_yaw
            [1, 0, 0],  # right_hip_roll
            [0, 1, 0],  # right_hip_pitch
            [0, 1, 0],  # right_knee
            [0, 1, 0],  # right_ankle
            [0, 0, 1],  # torso
            [0, 1, 0],  # left_shoulder_pitch
            [1, 0, 0],  # left_shoulder_roll
            [0, 0, 1],  # left_shoulder_yaw
            [0, 1, 0],  # left_elbow
            [0, 1, 0],  # right_shoulder_pitch
            [1, 0, 0],  # right_shoulder_roll
            [0, 0, 1],  # right_shoulder_yaw
            [0, 1, 0],  # right_elbow
        ]]).to(self.device)

        self.h1_joint_names = [
            'pelvis',
            'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
            'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
            'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link',
            'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'
        ]
        self.h1_joint_names_augment = self.h1_joint_names + ["left_hand_link", "right_hand_link"]
        self.h1_joint_pick = [
            'pelvis', "left_knee_link", "left_ankle_link", 'right_knee_link', 'right_ankle_link',
            "left_shoulder_roll_link", "left_elbow_link", "left_hand_link",
            "right_shoulder_roll_link", "right_elbow_link", "right_hand_link"
        ]
        self.smpl_joint_pick = [
            "Pelvis", "L_Knee", "L_Ankle", "R_Knee", "R_Ankle",
            "L_Shoulder", "L_Elbow", "L_Hand", "R_Shoulder", "R_Elbow", "R_Hand"
        ]
        self.h1_joint_pick_idx = [self.h1_joint_names_augment.index(j) for j in self.h1_joint_pick]
        self.smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in self.smpl_joint_pick]

    def retarget_motion(self, amass_data, shape_new):
        """Retarget AMASS motion data to H1 robot."""
        skip = int(amass_data['fps'] // self.config.FPS)
        trans = torch.from_numpy(amass_data['trans'][::skip]).float().to(self.device)
        N = trans.shape[0]
        pose_aa_walk = torch.from_numpy(np.concatenate((amass_data['pose_aa'][::skip, :66], np.zeros((N, 6))), axis=-1)).float().to(self.device)

        # Compute SMPL joints and root offset
        verts, joints = self.smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
        offset = joints[:, 0] - trans
        root_trans_offset = trans + offset

        # Initialize H1 pose
        pose_aa_h1 = np.repeat(np.repeat(sRot.identity().as_rotvec()[None, None, None, :], 22, axis=2), N, axis=1)
        pose_aa_h1[..., 0, :] = (sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()
        pose_aa_h1 = torch.from_numpy(pose_aa_h1).float().to(self.device)
        gt_root_rot = torch.from_numpy((sRot.from_rotvec(pose_aa_walk.cpu().numpy()[:, :3]) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_rotvec()).float().to(self.device)

        # Optimization setup
        dof_pos = torch.zeros((1, N, 19, 1)).to(self.device)
        dof_pos_new = Variable(dof_pos, requires_grad=True)
        optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=self.config.LR)

        best_loss = float('inf')
        patience_counter = 0

        # Optimization loop
        for iteration in range(self.config.MAX_ITERATIONS):
            verts, joints = self.smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
            pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], self.h1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(self.device)], axis=2).to(self.device)
            fk_return = self.h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])

            # Position loss
            diff = fk_return['global_translation_extend'][:, :, self.h1_joint_pick_idx] - joints[:, self.smpl_joint_pick_idx]
            pos_loss = diff.norm(dim=-1).mean()

            # Torque penalty
            torques = compute_torques(dof_pos_new, self.robot, self.data, self.device)
            torque_penalty = torch.clamp(torch.abs(torques) - self.h1_torque_limits[None, :], min=0).mean()

            # Joint limit penalty
            joint_min = self.h1_fk.joints_range[:, 0, None].to(self.device)
            joint_max = self.h1_fk.joints_range[:, 1, None].to(self.device)
            violation_min = torch.clamp(joint_min - dof_pos_new, min=0)
            violation_max = torch.clamp(dof_pos_new - joint_max, min=0)
            joint_limit_penalty = (violation_min ** 2 + violation_max ** 2).mean()

            # Total loss
            loss = (self.config.LOSS_WEIGHTS['pos'] * pos_loss +
                    self.config.LOSS_WEIGHTS['torque_penalty'] * torque_penalty +
                    self.config.LOSS_WEIGHTS['joint_limit_penalty'] * joint_limit_penalty)

            # Early stopping
            current_loss = loss.item()
            if current_loss < best_loss - self.config.EARLY_STOP_THRESHOLD:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.EARLY_STOP_PATIENCE:
                    print(f"Early stopping at iteration {iteration}")
                    break

            optimizer_pose.zero_grad()
            loss.backward()
            optimizer_pose.step()
            dof_pos_new.data.clamp_(joint_min, joint_max)

        # Final pose computation
        dof_pos_new.data.clamp_(self.h1_fk.joints_range[:, 0, None], self.h1_fk.joints_range[:, 1, None])
        pose_aa_h1_new = torch.cat([gt_root_rot[None, :, None], self.h1_rotation_axis * dof_pos_new, torch.zeros((1, N, 2, 3)).to(self.device)], axis=2)
        fk_return = self.h1_fk.fk_batch(pose_aa_h1_new, root_trans_offset[None, ])

        # Adjust root translation
        root_trans_offset_dump = root_trans_offset.clone()
        root_trans_offset_dump[..., 2] -= fk_return.global_translation[..., 2].min().item() - 0.08

        return {
            "root_trans_offset": root_trans_offset_dump.squeeze().cpu().detach().numpy(),
            "pose_aa": pose_aa_h1_new.squeeze().cpu().detach().numpy(),
            "dof": dof_pos_new.squeeze().detach().cpu().numpy(),
            "root_rot": sRot.from_rotvec(gt_root_rot.cpu().numpy()).as_quat(),
            "fps": self.config.FPS
        }

def load_shape_new(config):
    """Load shape parameters and move to device."""
    shape_new, scale = joblib.load(config.SHAPE_PATH)
    return shape_new.to(config.DEVICE)

def process_amass_file(data_key, data_path, config):
    """Process a single AMASS file and return retargeted data with its key."""
    retargeter = Retargeter(config)
    shape_new = load_shape_new(config)
    amass_data = load_amass_data(data_path)
    if "flag" in amass_data:
        print(f"Skipping {data_path}, missing framerate")
        return (data_key, None)
    retargeted_data = retargeter.retarget_motion(amass_data, shape_new)
    return (data_key, retargeted_data)

def process_wrapper(args):
    """Wrapper to unpack arguments for multiprocessing."""
    return process_amass_file(*args)

if __name__ == "__main__":
    # Initialize configuration
    config = RetargetConfig()

    # Process AMASS data
    amass_root = config.AMASS_ROOT
    all_pkls = glob.glob(f"{amass_root}/**/*.npz", recursive=True)
    split_len = len(amass_root.split("/"))
    key_name_to_pkls = {"0-" + "_".join(data_path.split("/")[split_len:]).replace(".npz", ""): data_path for data_path in all_pkls}

    if not key_name_to_pkls:
        raise ValueError(f"No motion files found in {amass_root}")

    # Prepare arguments for multiprocessing
    args_list = [(key, path, config) for key, path in key_name_to_pkls.items()]
    
    # Process files in parallel with progress bar
    data_dump = {}
    with mp.Pool(config.NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_wrapper, args_list),
            total=len(args_list),
            desc="Processing AMASS files"
        ))
        data_dump = {key: result for key, result in results if result is not None}

    # Save results
    joblib.dump(data_dump, config.SAVE_PATH)