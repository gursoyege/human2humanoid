import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_parser import SMPL_Parser
from phc.utils.torch_h1_humanoid_batch import Humanoid_Batch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Tuple
import joblib

# Visualization imports
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# SMPL joint names
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES

# H1 robot joint definitions
h1_joint_names = [
    'pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link', 'left_ankle_link',
    'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link', 'right_knee_link', 'right_ankle_link',
    'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link',
    'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'
]
h1_joint_names_augment = h1_joint_names + ["left_hand_link", "right_hand_link", "head_link"]

h1_joint_pick = [
    'pelvis', 'left_knee_link', "left_ankle_link", 'right_knee_link', 'right_ankle_link',
    "left_shoulder_roll_link", "left_elbow_link", "left_hand_link", "right_shoulder_roll_link",
    "right_elbow_link", "right_hand_link", "head_link"
]
smpl_joint_pick = [
    "Pelvis", "L_Knee", "L_Ankle", "R_Knee", "R_Ankle", "L_Shoulder", "L_Elbow", "L_Hand",
    "R_Shoulder", "R_Elbow", "R_Hand", "Head"
]

h1_joint_pick_idx = [h1_joint_names_augment.index(j) for j in h1_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

# Configuration class
class ShapeOptConfig:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_ITER = 1000
    MJCF_PATH = "resources/robots/h1/h1.xml"
    LR = 0.1
    LOSS_WEIGHTS = {'pos': 1.0, 'beta_reg': 0.00, 'com': 0.0}  # Updated weights
    RATE_PATIENCE = 50  # For learning rate scheduler
    NUM_POSES = 4
    BATCH_SIZE = NUM_POSES
    EARLY_STOPPING_PATIENCE = 20  # Number of iterations to wait without improvement
    EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum improvement in loss to reset patience

class ShapeOptimizer:
    def __init__(self, config: ShapeOptConfig):
        self.config = config
        self.config.BATCH_SIZE = self.config.NUM_POSES  # Set BATCH_SIZE to NUM_POSES
        self._init_models()
        self._setup_optimization()

    def _init_models(self) -> None:
        self.h1_fk = Humanoid_Batch(mjcf_file=self.config.MJCF_PATH, extend_hand=True, extend_head=True, device=self.config.DEVICE)
        self.smpl_parser = SMPL_Parser(model_path="data/smpl", gender="neutral").to(self.config.DEVICE)
        self.betas = torch.zeros((1, 10), device=self.config.DEVICE, requires_grad=True)
        self.scale = torch.ones((1, 3), device=self.config.DEVICE, requires_grad=True)
        self.global_trans = torch.zeros((1, 3), device=self.config.DEVICE, requires_grad=True)
        self.h1_joint_idx = torch.tensor(h1_joint_pick_idx, device=self.config.DEVICE)
        self.smpl_joint_idx = torch.tensor(smpl_joint_pick_idx, device=self.config.DEVICE)
        self.pos_weights = torch.ones(len(h1_joint_pick_idx), device=self.config.DEVICE)
        
        self.h1_masses = torch.tensor([
            5.39, 1.721, 0.474, 1.721, 0.474, 0.793, 0.723, 0.5, 0.793, 0.723, 0.5, 3.0
        ], device=self.config.DEVICE)
        
        self.smpl_mass_ratios = torch.tensor([
            0.43, 0.045, 0.014, 0.045, 0.014, 0.025, 0.015, 0.006, 0.025, 0.015, 0.006, 0.07
        ], device=self.config.DEVICE)

    def visualize_smpl_poses(self, betas=None, global_trans=None):
        smpl_poses = self._get_smpl_poses().to(self.config.DEVICE)
        num_poses = smpl_poses.shape[0]
        if betas is None:
            betas = torch.zeros(num_poses, 10, device=self.config.DEVICE)
        if global_trans is None:
            global_trans = torch.zeros(num_poses, 3, device=self.config.DEVICE)

        smpl_verts, smpl_joints = self.smpl_parser.get_joints_verts(
            smpl_poses, betas.to(self.config.DEVICE), global_trans.to(self.config.DEVICE)
        )
        smpl_faces = self.smpl_parser.faces_tensor.to(self.config.DEVICE)

        fig = make_subplots(
            rows=1, cols=num_poses,
            subplot_titles=[f"Pose {i+1}" for i in range(num_poses)],
            specs=[[{'type': 'scene'}] * num_poses]
        )

        for i in range(num_poses):
            smpl_mesh = Meshes(verts=[smpl_verts[i].detach()], faces=[smpl_faces.detach()])
            mesh_plot = go.Mesh3d(
                x=smpl_mesh.verts_list()[0][:, 0].detach().cpu().numpy(),
                y=smpl_mesh.verts_list()[0][:, 1].detach().cpu().numpy(),
                z=smpl_mesh.verts_list()[0][:, 2].detach().cpu().numpy(),
                i=smpl_mesh.faces_list()[0][:, 0].detach().cpu().numpy(),
                j=smpl_mesh.faces_list()[0][:, 1].detach().cpu().numpy(),
                k=smpl_mesh.faces_list()[0][:, 2].detach().cpu().numpy(),
                opacity=0.5,
                color='lightblue'
            )
            fig.add_trace(mesh_plot, row=1, col=i+1)

            joints_plot = go.Scatter3d(
                x=smpl_joints[i, :, 0].detach().cpu().numpy(),
                y=smpl_joints[i, :, 1].detach().cpu().numpy(),
                z=smpl_joints[i, :, 2].detach().cpu().numpy(),
                mode='markers',
                marker=dict(size=5, color='red')
            )
            fig.add_trace(joints_plot, row=1, col=i+1)

            fig.update_scenes(
                dict(
                    aspectmode='data',
                    camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.25, y=1.25, z=0.5))
                ),
                row=1, col=i+1
            )

        fig.update_layout(
            width=300 * num_poses,
            height=600,
            title_text="SMPL Poses Visualization"
        )
        fig.write_html("smpl_poses_visualization.html")
        print("Visualization saved as 'smpl_poses_visualization.html'")

    def visualize_h1_poses(self):
        h1_poses = self._get_h1_poses().to(self.config.DEVICE)
        num_poses = h1_poses.shape[0]
        
        fk_return = self.h1_fk.fk_batch(h1_poses, torch.zeros((num_poses, 1, 3), device=self.config.DEVICE))
        h1_joints = fk_return.global_translation_extend.squeeze(1)
        
        skeletal_connections = [
            (h1_joint_names_augment.index('pelvis'), h1_joint_names_augment.index('torso_link')),
            (h1_joint_names_augment.index('torso_link'), h1_joint_names_augment.index('head_link')),
            (h1_joint_names_augment.index('pelvis'), h1_joint_names_augment.index('left_hip_yaw_link')),
            (h1_joint_names_augment.index('left_hip_yaw_link'), h1_joint_names_augment.index('left_hip_roll_link')),
            (h1_joint_names_augment.index('left_hip_roll_link'), h1_joint_names_augment.index('left_hip_pitch_link')),
            (h1_joint_names_augment.index('left_hip_pitch_link'), h1_joint_names_augment.index('left_knee_link')),
            (h1_joint_names_augment.index('left_knee_link'), h1_joint_names_augment.index('left_ankle_link')),
            (h1_joint_names_augment.index('pelvis'), h1_joint_names_augment.index('right_hip_yaw_link')),
            (h1_joint_names_augment.index('right_hip_yaw_link'), h1_joint_names_augment.index('right_hip_roll_link')),
            (h1_joint_names_augment.index('right_hip_roll_link'), h1_joint_names_augment.index('right_hip_pitch_link')),
            (h1_joint_names_augment.index('right_hip_pitch_link'), h1_joint_names_augment.index('right_knee_link')),
            (h1_joint_names_augment.index('right_knee_link'), h1_joint_names_augment.index('right_ankle_link')),
            (h1_joint_names_augment.index('torso_link'), h1_joint_names_augment.index('left_shoulder_pitch_link')),
            (h1_joint_names_augment.index('left_shoulder_pitch_link'), h1_joint_names_augment.index('left_shoulder_roll_link')),
            (h1_joint_names_augment.index('left_shoulder_roll_link'), h1_joint_names_augment.index('left_shoulder_yaw_link')),
            (h1_joint_names_augment.index('left_shoulder_yaw_link'), h1_joint_names_augment.index('left_elbow_link')),
            (h1_joint_names_augment.index('left_elbow_link'), h1_joint_names_augment.index('left_hand_link')),
            (h1_joint_names_augment.index('torso_link'), h1_joint_names_augment.index('right_shoulder_pitch_link')),
            (h1_joint_names_augment.index('right_shoulder_pitch_link'), h1_joint_names_augment.index('right_shoulder_roll_link')),
            (h1_joint_names_augment.index('right_shoulder_roll_link'), h1_joint_names_augment.index('right_shoulder_yaw_link')),
            (h1_joint_names_augment.index('right_shoulder_yaw_link'), h1_joint_names_augment.index('right_elbow_link')),
            (h1_joint_names_augment.index('right_elbow_link'), h1_joint_names_augment.index('right_hand_link')),
        ]
        
        fig = make_subplots(
            rows=1, cols=num_poses,
            subplot_titles=[f"Pose {i+1}" for i in range(num_poses)],
            specs=[[{'type': 'scene'}] * num_poses]
        )
        
        for i in range(num_poses):
            joints_plot = go.Scatter3d(
                x=h1_joints[i, :, 0].detach().cpu().numpy(),
                y=h1_joints[i, :, 1].detach().cpu().numpy(),
                z=h1_joints[i, :, 2].detach().cpu().numpy(),
                mode='markers',
                marker=dict(size=5, color='green'),
                name=f'Pose {i+1} Joints'
            )
            fig.add_trace(joints_plot, row=1, col=i+1)
            
            for start_idx, end_idx in skeletal_connections:
                line_plot = go.Scatter3d(
                    x=[h1_joints[i, start_idx, 0].detach().cpu().numpy(), h1_joints[i, end_idx, 0].detach().cpu().numpy()],
                    y=[h1_joints[i, start_idx, 1].detach().cpu().numpy(), h1_joints[i, end_idx, 1].detach().cpu().numpy()],
                    z=[h1_joints[i, start_idx, 2].detach().cpu().numpy(), h1_joints[i, end_idx, 2].detach().cpu().numpy()],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    name=f'Pose {i+1} Skeleton'
                )
                fig.add_trace(line_plot, row=1, col=i+1)
            
            picked_joints = h1_joints[i, self.h1_joint_idx]
            picked_plot = go.Scatter3d(
                x=picked_joints[:, 0].detach().cpu().numpy(),
                y=picked_joints[:, 1].detach().cpu().numpy(),
                z=picked_joints[:, 2].detach().cpu().numpy(),
                mode='markers',
                marker=dict(size=7, color='red'),
                name=f'Key Joints {i+1}'
            )
            fig.add_trace(picked_plot, row=1, col=i+1)
            
            fig.update_scenes(
                dict(
                    aspectmode='data',
                    camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.25, y=1.25, z=0.5)),
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            width=300 * num_poses,
            height=600,
            title_text="H1 Poses Visualization with Human-like Skeleton",
            showlegend=True
        )
        fig.write_html("h1_poses_visualization.html")
        print("Visualization saved as 'h1_poses_visualization.html'")

    def _setup_optimization(self) -> None:
        self.optimizer = Adam([self.betas, self.scale, self.global_trans], lr=self.config.LR)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=self.config.RATE_PATIENCE, verbose=True)

    def _get_smpl_poses(self) -> torch.Tensor:
        device, num_poses = self.config.DEVICE, self.config.NUM_POSES
        poses = torch.zeros((num_poses, 72), device=device)
        # Helper conversion: set mode='quat' for quaternion conversion
        conv = lambda angles, mode='euler': torch.from_numpy(
            (sRot.from_quat(angles).as_rotvec() if mode == 'quat'
            else sRot.from_euler("xyz", angles, degrees=False).as_rotvec())
        ).float().to(device)

        # ----- Pose 0 -----
        # Base rotation from a quaternion (applied to the first 3 entries)
        poses[0, :3] = conv([0.5, 0.5, 0.5, 0.5], mode='quat')
        for bone, ang in {
            'L_Shoulder': [0, 0, -np.pi / 2],
            'R_Shoulder': [0, 0,  np.pi / 2],
            'L_Elbow'   : [0, -np.pi / 2, 0],
            'R_Elbow'   : [0,  np.pi / 2, 0]
        }.items():
            idx = SMPL_BONE_ORDER_NAMES.index(bone) * 3
            poses[0, idx:idx + 3] = conv(ang)

        # ----- Pose 1 -----
        poses[1] = poses[0].clone()
        for bone, ang in {
            'L_Hip'     : [-np.pi / 3, 0, 0],
            'R_Hip'     : [-np.pi / 3, 0, 0],
            'L_Knee'    : [ np.pi / 2, 0, 0],
            'R_Knee'    : [ np.pi / 2, 0, 0],
            'L_Ankle'   : [-np.pi / 6, 0, 0],
            'R_Ankle'   : [-np.pi / 6, 0, 0],
            'R_Shoulder': [0, np.pi / 2, 0],
            'L_Shoulder': [0, -np.pi / 2, 0],
            'L_Elbow'   : [0, 0, 0],
            'R_Elbow'   : [0, 0, 0]
        }.items():
            idx = SMPL_BONE_ORDER_NAMES.index(bone) * 3
            poses[1, idx:idx + 3] = conv(ang)

        # ----- Pose 2 -----
        poses[2] = poses[0].clone()
        for bone, ang in {
            'R_Shoulder': [0, 0, -np.pi / 6],
            'L_Shoulder': [0, 0,  np.pi / 6],
            'L_Hip'     : [0, 0,  np.pi / 6],
            'R_Hip'     : [0, 0, -np.pi / 6],
            'R_Ankle'   : [0, 0,  np.pi / 6],
            'L_Ankle'   : [0, 0, -np.pi / 6],
            'L_Elbow'   : [0, 0, 0],
            'R_Elbow'   : [0, 0, 0]
        }.items():
            idx = SMPL_BONE_ORDER_NAMES.index(bone) * 3
            poses[2, idx:idx + 3] = conv(ang)

        # ----- Pose 3 -----
        poses[3] = poses[2].clone()
        for bone, ang in {
            'R_Shoulder': [0, 0, 0],
            'L_Shoulder': [0, 0, 0],
            'L_Hip'     : [-1.5 * np.pi / 3, 0, 0],
            'R_Hip'     : [-1.5 * np.pi / 3, 0, 0],
            'L_Knee'    : [1.5 * np.pi / 2, 0, 0],
            'R_Knee'    : [1.5 * np.pi / 2, 0, 0],
            'L_Ankle'   : [-1.5 * np.pi / 6, 0, 0],
            'R_Ankle'   : [-1.5 * np.pi / 6, 0, 0]
        }.items():
            idx = SMPL_BONE_ORDER_NAMES.index(bone) * 3
            poses[3, idx:idx + 3] = conv(ang)

        return poses


    def _get_h1_poses(self) -> torch.Tensor:
        device = self.config.DEVICE
        num_joints = len(self.h1_fk._parents) - 1
        poses = torch.zeros((self.config.NUM_POSES, 1, num_joints, 3), device=device)
        conv = lambda ang: torch.from_numpy(
            sRot.from_euler("xyz", ang, degrees=False).as_rotvec()
        ).float().to(device)

        # Helper to assign rotation to a joint in the pose tensor
        def set_h1(pose_idx, joint, angles):
            idx = h1_joint_names.index(joint)
            poses[pose_idx, 0, idx] = conv(angles)

        # ----- Pose 1 -----
        for joint, ang in {
            'left_hip_pitch_link'    : [0, -np.pi / 3, 0],
            'right_hip_pitch_link'   : [0, -np.pi / 3, 0],
            'left_knee_link'         : [0,  np.pi / 2, 0],
            'right_knee_link'        : [0,  np.pi / 2, 0],
            'left_elbow_link'        : [0,  np.pi / 2, 0],
            'right_elbow_link'       : [0,  np.pi / 2, 0],
            'left_shoulder_roll_link': [0, -np.pi / 2, 0],
            'right_shoulder_roll_link':[0, -np.pi / 2, 0],
            'left_ankle_link'        : [0, -np.pi / 6, 0],
            'right_ankle_link'       : [0, -np.pi / 6, 0]
        }.items():
            set_h1(1, joint, ang)

        # ----- Pose 2 -----
        for joint, ang in {
            'left_elbow_link'        : [0,  np.pi / 2, 0],
            'right_elbow_link'       : [0,  np.pi / 2, 0],
            'left_hip_pitch_link'    : [np.pi / 6, 0, 0],
            'right_hip_pitch_link'   : [-np.pi / 6, 0, 0],
            'left_shoulder_roll_link': [np.pi / 2 + np.pi / 6, 0, 0],
            'right_shoulder_roll_link': [-np.pi / 2 - np.pi / 6, 0, 0],
            'left_ankle_link'        : [-np.pi / 6, 0, 0],
            'right_ankle_link'       : [np.pi / 6, 0, 0]
        }.items():
            set_h1(2, joint, ang)

        # ----- Pose 3 -----
        for joint, ang in {
            'left_elbow_link'        : [0,  np.pi / 2, 0],
            'right_elbow_link'       : [0,  np.pi / 2, 0],
            'left_hip_pitch_link'    : [0, -1.5 * np.pi / 3, 0],
            'right_hip_pitch_link'   : [0, -1.5 * np.pi / 3, 0],
            'left_knee_link'         : [0,  1.5 * np.pi / 2, 0],
            'right_knee_link'        : [0,  1.5 * np.pi / 2, 0],
            'left_shoulder_roll_link': [np.pi / 2, 0, 0],
            'right_shoulder_roll_link':[ -np.pi / 2, 0, 0],
            'left_ankle_link'        : [0, -1.5 * np.pi / 6, 0],
            'right_ankle_link'       : [0, -1.5 * np.pi / 3, 0]
        }.items():
            set_h1(3, joint, ang)

        return poses


    def optimize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        smpl_poses = self._get_smpl_poses()
        self.visualize_smpl_poses()
        h1_poses = self._get_h1_poses()
        self.visualize_h1_poses()        
        fk_return = self.h1_fk.fk_batch(h1_poses, torch.zeros((self.config.NUM_POSES, 1, 3), device=self.config.DEVICE))
        h1_trans = fk_return.global_translation_extend.squeeze(1)

        trans_optimizer = torch.optim.Adam([self.global_trans], lr=0.1)
        trans_best_loss = float('inf')
        trans_patience_counter = 0

        for _ in range(100):
            trans_optimizer.zero_grad()
            smpl_verts, smpl_joints = self.smpl_parser.get_joints_verts(
                smpl_poses, self.betas.expand(self.config.NUM_POSES, -1), torch.zeros_like(smpl_poses[:, :3])
            )
            smpl_joints = smpl_joints + self.global_trans.unsqueeze(1)
            pos_diff = (h1_trans[:, self.h1_joint_idx] - smpl_joints[:, self.smpl_joint_idx]).norm(p=2, dim=-1)
            pos_loss = pos_diff.mean()
            pos_loss.backward()
            trans_optimizer.step()

            if pos_loss.item() < trans_best_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                trans_best_loss = pos_loss.item()
                trans_patience_counter = 0
            else:
                trans_patience_counter += 1

            if trans_patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print("Early stopping in translation optimization")
                break

        print("--- After Translation Optimization ---")
        print("Optimized translation:", self.global_trans[0].detach().cpu().numpy())
        print("Average distance:", pos_diff.mean().item())

        self.scale = torch.ones((1, 3), device=self.config.DEVICE, requires_grad=True)
        scale_optimizer = torch.optim.Adam([self.scale], lr=0.01)
        scale_best_loss = float('inf')
        scale_patience_counter = 0

        for _ in range(100):
            scale_optimizer.zero_grad()
            smpl_verts, smpl_joints = self.smpl_parser.get_joints_verts(
                smpl_poses, self.betas.expand(self.config.NUM_POSES, -1), torch.zeros_like(smpl_poses[:, :3])
            )
            root_pos = smpl_joints[:, 0:1, :]
            smpl_joints = root_pos + (smpl_joints - root_pos) * self.scale
            smpl_joints = smpl_joints + self.global_trans.unsqueeze(1)
            pos_diff = (h1_trans[:, self.h1_joint_idx] - smpl_joints[:, self.smpl_joint_idx]).norm(p=2, dim=-1)
            pos_loss = pos_diff.mean()
            pos_loss.backward()
            scale_optimizer.step()

            if pos_loss.item() < scale_best_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                scale_best_loss = pos_loss.item()
                scale_patience_counter = 0
            else:
                scale_patience_counter += 1

            if scale_patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print("Early stopping in scale optimization")
                break

        print("--- After Scale Optimization ---")
        print(f"Optimized scale: X={self.scale[0,0].item():.3f}, Y={self.scale[0,1].item():.3f}, Z={self.scale[0,2].item():.3f}")
        print("Average distance:", pos_diff.mean().item())

        best_loss = float('inf')
        patience_counter = 0

        for i in tqdm(range(self.config.NUM_ITER), desc="Full Optimization"):
            self.optimizer.zero_grad()
            smpl_verts, smpl_joints = self.smpl_parser.get_joints_verts(
                smpl_poses, self.betas.expand(self.config.NUM_POSES, -1), torch.zeros_like(smpl_poses[:, :3])
            )
            root_pos = smpl_joints[:, 0:1, :]
            smpl_joints = root_pos + (smpl_joints - root_pos) * self.scale
            smpl_joints = smpl_joints + self.global_trans.unsqueeze(1)
            pos_diff = (h1_trans[:, self.h1_joint_idx] - smpl_joints[:, self.smpl_joint_idx]).norm(p=2, dim=-1)
            pos_loss = pos_diff.mean()

            com_h1 = (self.h1_masses[None, :, None] * h1_trans[:, self.h1_joint_idx]).sum(dim=1) / self.h1_masses.sum()
            com_smpl = (self.smpl_mass_ratios[None, :, None] * smpl_joints[:, self.smpl_joint_idx]).sum(dim=1) / self.smpl_mass_ratios.sum()
            com_loss = (com_h1 - com_smpl).norm(dim=-1).mean()

            beta_reg_loss = (self.betas ** 2).mean()

            total_loss = (
                self.config.LOSS_WEIGHTS['pos'] * pos_loss +
                self.config.LOSS_WEIGHTS['com'] * com_loss +
                self.config.LOSS_WEIGHTS['beta_reg'] * beta_reg_loss
            )
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step(total_loss)

            if total_loss.item() < best_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at iteration {i}")
                break

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {total_loss.item():.3f}, "
                      f"Scale X={self.scale[0,0].item():.3f}, Y={self.scale[0,1].item():.3f}, Z={self.scale[0,2].item():.3f}, "
                      f"Pos Loss = {pos_loss.item():.3f}, COM Loss = {com_loss.item():.3f}, "
                      f"Beta Reg Loss = {beta_reg_loss.item():.3f}")

        print("--- Final Results ---")
        print(f"Optimized scale: X={self.scale[0,0].item():.3f}, Y={self.scale[0,1].item():.3f}, Z={self.scale[0,2].item():.3f}")
        print("Optimized translation:", self.global_trans[0].detach().cpu().numpy())
        print("Final average distance:", pos_loss.item())
        print("Final COM loss:", com_loss.item())
        print("Final beta reg loss:", beta_reg_loss.item())

        smpl_faces = self.smpl_parser.faces_tensor
        self._visualize_results(smpl_verts, h1_trans, smpl_faces)

        os.makedirs("data/h1", exist_ok=True)
        save_path = "data/h1/shape_optimized_v3.pkl"
        joblib.dump({
            self.betas.detach().cpu(),
            self.scale.detach().cpu(),
        }, save_path)
        print(f"Optimized parameters saved to {save_path}")

        return self.betas, self.scale, self.global_trans

    def _visualize_results(self, smpl_verts, h1_joints, smpl_faces):
        smpl_verts = smpl_verts.to(self.config.DEVICE)
        root_pos = smpl_verts[:, 0:1, :]
        smpl_verts = root_pos + (smpl_verts - root_pos) * self.scale
        smpl_verts = smpl_verts + self.global_trans.unsqueeze(1)
        smpl_verts = smpl_verts.cpu()
        h1_joints = h1_joints.cpu()
        smpl_faces = smpl_faces.cpu()

        fig = make_subplots(rows=1, cols=self.config.NUM_POSES, 
                            subplot_titles=[f"Pose {i+1}" for i in range(self.config.NUM_POSES)],
                            specs=[[{'type': 'scene'}] * self.config.NUM_POSES])

        for i in range(self.config.NUM_POSES):
            smpl_mesh = Meshes(verts=[smpl_verts[i]], faces=[smpl_faces])
            mesh_plot = plot_scene({"scene": {"SMPL Mesh": smpl_mesh}}).data[0]
            mesh_plot.update(opacity=0.5)
            fig.add_trace(mesh_plot, row=1, col=i+1)

            h1_picked_joints = h1_joints[i, self.h1_joint_idx.cpu()]
            h1_points = Pointclouds(points=[h1_picked_joints])
            joints_plot = plot_scene({"scene": {"H1 Joints": h1_points}}).data[0]
            joints_plot.marker.size = 10
            fig.add_trace(joints_plot, row=1, col=i+1)

            fig.update_scenes(dict(
                aspectmode='data',
                camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=1.25, y=1.25, z=0.5))
            ), row=1, col=i+1)

        fig.update_layout(
            width=300 * self.config.NUM_POSES,
            height=600,
            title_text="SMPL and H1 Joints Across All Poses"
        )
        fig.write_html("shape_fitting_result.html")
        print("Visualization saved as 'shape_fitting_result.html'")

if __name__ == "__main__":
    config = ShapeOptConfig()
    optimizer = ShapeOptimizer(config)
    betas, scale, global_trans = optimizer.optimize()