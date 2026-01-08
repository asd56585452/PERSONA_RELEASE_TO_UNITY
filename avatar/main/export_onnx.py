import argparse
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.ops import knn_points
import numpy as np
import json
import cv2
from glob import glob
from tqdm import tqdm
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from utils.preprocessing import set_aspect_ratio, get_patch_image
from plyfile import PlyData, PlyElement

def rgb_to_sh(rgb):
    """
    將 [0, 1] 範圍的 RGB 顏色轉換為 0 階球諧函數 (Spherical Harmonics) 係數。
    這是 3DGS 論文中 SH2RGB 的逆操作。 C0 = 0.28209479177387814
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

# --- 修改後的儲存函式 ---
def save_to_ply_3dgs_format(path, mean_3d, rgb, raw_opacity, raw_scale, rotation):
    """
    將模型參數儲存為原始 3DGS 論文中定義的 .ply 格式。
    這個格式儲存的是未經啟動函數處理的原始可訓練參數。

    Args:
        path (str): 儲存 .ply 檔案的路徑.
        mean_3d (np.array): Gaussians 的中心點位置 (N, 3).
        rgb (np.array): Gaussians 的顏色 (N, 3), 範圍應在 [0, 1].
        raw_opacity (np.array): 未經 sigmoid 處理的原始 logit opacity (N, 1).
        raw_scale (np.array): 未經 exp() 處理的原始 log-space scale (N, 3).
        rotation (np.array): 旋轉四元數 (N, 4).
    """
    print(f"正在將輸出儲存為原始 3DGS 格式至 '{path}'...")
    
    num_points = mean_3d.shape[0]

    # 1. 處理顏色: 將 RGB 轉換為 f_dc (0階 SH)。f_rest 設為 0。
    features_dc = rgb_to_sh(rgb).astype(np.float32)
    # 假設 SH degree 為 3，則 rest features 有 15*3=45 個
    # 如果您的模型不需要這麼高的階數，可以設為更小的值，但為了兼容性，這裡使用 45
    features_rest = np.zeros((num_points, 0), dtype=np.float32)

    # 確保其他參數是 float32
    xyz = mean_3d.astype(np.float32)
    opacity = np.clip(raw_opacity, 1e-6, 1.0 - 1e-6)
    opacities = np.log(opacity / (1.0 - opacity))
    scales = np.log(raw_scale).astype(np.float32)
    rotations = rotation.astype(np.float32)
    
    # 建立所有屬性的列表
    dtype_full = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
    ]
    # 添加 f_rest 屬性
    for i in range(features_rest.shape[1]):
        dtype_full.append((f'f_rest_{i}', 'f4'))
    
    dtype_full.extend([
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')
    ])

    # 準備寫入檔案的資料
    elements = np.empty(num_points, dtype=dtype_full)
    normals = np.zeros_like(xyz, dtype=np.float32)
    
    attributes = np.concatenate((
        xyz,
        normals,
        features_dc,
        features_rest,
        opacities,
        scales,
        rotations
    ), axis=1)
    
    elements[:] = list(map(tuple, attributes))

    # 建立 PlyData 物件並寫入檔案
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    
    print(f"✅ 成功以原始 3DGS 格式儲存 {num_points} 個 Gaussians 至 '{path}'。")

class ModelWrapper(torch.nn.Module):
    def __init__(self, model,smplx_keys,cam_keys,smplx_param, cam_param,mode=None):
        super().__init__()
        self.model = model
        self.mode = mode
        if mode=="full":
            self.output_names = [
                    'mean_3d',
                        'opacity',
                        'scale',
                        'rotation', 
                        'rgb',
                        'mean_3d_refined',
                        # 'skinning_weight_refined',
                        'joint_zero_pose',
                        'transform_mat_neutral_pose',
                        'parents',
                        'skinning_weight'
                ]
        elif mode=="no_refine":
            self.output_names = [
                    'mean_3d'
                ]
        elif mode=="refine":
            self.output_names = [
                    'mean_3d_refined',
                    # 'skinning_weight_refined'
                ]
        elif mode=="static":
            self.output_names = [
                'scale',
                        'rgb',
                    'joint_zero_pose',
                        'transform_mat_neutral_pose',
                        'parents',
                        'skinning_weight'
                ]
        with torch.no_grad():
            vert_neutral_pose, vert_neutral_pose_wo_upsample  = model.module.persona.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
            joint_zero_pose = model.module.persona.get_zero_pose_human()
            # get geometry Gaussian features
            mean_offset = model.module.persona.mean_offset
            scale = torch.exp(model.module.persona.scale).repeat(1,3)
            rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(smpl_x.vertex_num_upsampled,1,1)) # constant
            opacity = torch.ones((smpl_x.vertex_num_upsampled,1)).float().cuda() # constant
            rgb = torch.sigmoid(model.module.persona.rgb)
            mean_3d = vert_neutral_pose + mean_offset # 大 pose

            # get skinning weight
            skinning_weight = model.module.persona.get_skinning_weight(mean_3d, scale.detach())

            # pose-dependent mean offsets
            tri_feat = model.module.persona.extract_tri_feature()

            # forward kinematics and lbs
            transform_mat_joint,transform_mat_neutral_pose = model.module.persona.get_transform_mat_joint(joint_zero_pose, smplx_param, jaw_zero_pose=True) # follow jaw_pose of the vert_neutral_pose
        self.register_buffer('tri_feat', tri_feat)
        self.register_buffer('scale', scale)
        self.register_buffer('rgb', rgb)
        self.register_buffer('mean_3d', mean_3d)
        self.register_buffer('joint_zero_pose', joint_zero_pose)
        self.register_buffer('transform_mat_neutral_pose', transform_mat_neutral_pose)
        self.register_buffer('parents', model.module.persona.smplx_layer.parents)
        self.register_buffer('skinning_weight', skinning_weight)
        self.register_buffer('vert_neutral_pose', vert_neutral_pose)
        self.register_buffer('mean_offset',mean_offset)
        self.register_buffer('opacity',opacity)
        self.register_buffer('rotation',rotation)
        self.smplx_keys = list(smplx_keys)
        self.cam_keys = list(cam_keys)

    def forward(self, *inputs):

        smplx_param = {}
        cam_param = {}
        smplx_input_count = len(self.smplx_keys)
        cam_inputs_count = len(self.cam_keys)
        smplx_inputs_tuple = inputs[:smplx_input_count]
        cam_inputs_tuple = inputs[smplx_input_count:smplx_input_count+cam_inputs_count]
        # joint_zero_pose = self.model.module.human_gaussian.get_zero_pose_human()
        for i, key in enumerate(self.smplx_keys):
            smplx_param[key] = smplx_inputs_tuple[i]
            
        for i, key in enumerate(self.cam_keys):
            cam_param[key] = cam_inputs_tuple[i]

        joint_idxs = torch.argmax(self.skinning_weight,1)
        mean_offset_offset = self.model.module.persona.get_mean_offset_offset(self.tri_feat, smplx_param, joint_idxs)
        mean_3d_refined = self.mean_3d + mean_offset_offset # 大 pose

        # smplx facial expression offset
        smplx_expr_offset = (smplx_param['expr'][None,None,:] * self.model.module.persona.expr_dirs).sum(2)
        vert = self.vert_neutral_pose + smplx_expr_offset # 大 pose
        mean_3d = self.mean_3d + smplx_expr_offset # 大 pose
        mean_3d_refined = mean_3d_refined + smplx_expr_offset # 大 pose

        # get skinning weight
        skinning_weight_refined = self.model.module.persona.get_skinning_weight(mean_3d_refined, self.scale.detach())

        if self.mode == "full":
            return (
                mean_3d,
                self.opacity,
                self.scale,
                self.rotation, 
                self.rgb,
                mean_3d_refined,
                # skinning_weight_refined,
                self.joint_zero_pose,
                self.transform_mat_neutral_pose,
                # self.nn_vertex_idxs,
                self.parents,
                self.skinning_weight
            )
        elif self.mode == "no_refine":
            return (
                mean_3d,
            )
        elif self.mode == "refine":
            return (
                mean_3d_refined,
                # skinning_weight_refined
            )
        elif self.mode == "static":
            return (
                self.scale,
                self.rgb,
                self.joint_zero_pose,
                self.transform_mat_neutral_pose,
                self.parents,
                self.skinning_weight
            )

        # forward kinematics and lbs
        transform_mat_joint,_ = self.model.module.persona.get_transform_mat_joint(self.joint_zero_pose, smplx_param, jaw_zero_pose=True) # follow jaw_pose of the vert_neutral_pose
        transform_mat_vertex = torch.matmul(self.skinning_weight, transform_mat_joint.view(smpl_x.joint['num'],16)).view(smpl_x.vertex_num_upsampled,4,4)
        vert = self.model.module.persona.lbs(vert, transform_mat_vertex) # posed with smplx_param
        mean_3d = self.model.module.persona.lbs(mean_3d, transform_mat_vertex) # posed with smplx_param
        transform_mat_vertex = torch.matmul(skinning_weight_refined, transform_mat_joint.view(smpl_x.joint['num'],16)).view(smpl_x.vertex_num_upsampled,4,4)
        mean_3d_refined = self.model.module.persona.lbs(mean_3d_refined, transform_mat_vertex) # posed with smplx_param

        # camera coordinate system -> world coordinate system
        vert = torch.matmul(torch.inverse(cam_param['R']), (vert - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        mean_3d = torch.matmul(torch.inverse(cam_param['R']), (mean_3d - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)
        mean_3d_refined = torch.matmul(torch.inverse(cam_param['R']), (mean_3d_refined - cam_param['t'].view(1,3)).permute(1,0)).permute(1,0)

        # Gaussians and offsets
        assets = {'mean_3d': mean_3d, 'opacity': self.opacity, 'scale': self.scale, 'rotation': self.rotation, 'rgb': self.rgb}
        assets_refined = {'mean_3d': mean_3d_refined, 'opacity': self.opacity, 'scale': self.scale, 'rotation': self.rotation, 'rgb': self.rgb}
        offsets = {'mean_offset': self.mean_offset, 'mean_offset_offset': mean_offset_offset}
        
        return assets, assets_refined, offsets, self.vert_neutral_pose, vert

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    parser.add_argument('--motion_path', type=str, dest='motion_path')
    parser.add_argument('--output_path', type=str, dest='output_path')
    parser.add_argument('--use_bkg', dest='use_bkg', action='store_true')
    args = parser.parse_args()

    assert args.subject_id, "Please set subject ID"
    assert args.test_epoch, 'Test epoch is required.'
    assert args.motion_path, 'Motion path for the animation is required.'
    return args

def main():
    # argument parse and create log
    args = parse_args()
    cfg.set_args(args.subject_id)

    tester = Tester(args.test_epoch)

    # set dummy data, which will be replaced with data from checkpoint
    root_path = osp.join('..', 'data', 'subjects', cfg.subject_id)
    smpl_x.set_id_info(None, None, None)
    smpl_x.set_texture(None, None, None)
    tester._make_model()
    model = tester.model.module

    motion_path = args.motion_path
    if motion_path[-1] == '/':
        motion_name = motion_path[:-1].split('/')[-1]
    else:        
        motion_name = motion_path.split('/')[-1]
    frame_idx_list = sorted([int(x.split('/')[-1][:-5]) for x in glob(osp.join(args.motion_path, 'smplx', 'params', '*.json'))])
    render_shape = cv2.imread(osp.join(args.motion_path, 'images', str(frame_idx_list[0]) + '.png')).shape[:2]
   
    # load reference image
    img_ref = cv2.imread(osp.join(osp.join('..', 'data', 'subjects', cfg.subject_id, 'captured', 'images', '0.png')))
    bbox = set_aspect_ratio(np.array([0,0,img_ref.shape[1],img_ref.shape[0]], dtype=np.float32), extend_ratio=1.5, aspect_ratio=img_ref.shape[1]/img_ref.shape[0])
    img_ref, _, _ = get_patch_image(img_ref, bbox, (render_shape[0], int(img_ref.shape[1]/img_ref.shape[0]*render_shape[0])), bordervalue=(255,255,255))
    frame_idx = frame_idx_list[200]

    with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
        cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
    with open(osp.join(args.motion_path, 'smplx', 'params', str(frame_idx) + '.json')) as f:
        smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}

    # forward
    with torch.no_grad():
        asset, asset_refined, offset, vert_neutral_pose, vert = model.persona(smplx_param, cam_param)
    
    tester.model.eval()

    frame_idx = frame_idx_list[0]
    with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
        cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
    with open(osp.join(args.motion_path, 'smplx', 'params', str(frame_idx) + '.json')) as f:
        smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}
    wrapped_model = ModelWrapper(tester.model,smplx_param.keys(),cam_param.keys(),smplx_param, cam_param).cuda().eval()

    frame_idx = frame_idx_list[200]
    with open(osp.join(args.motion_path, 'cam_params', str(frame_idx) + '.json')) as f:
        cam_param = {k: torch.FloatTensor(v).cuda() for k,v in json.load(f).items()}
    with open(osp.join(args.motion_path, 'smplx', 'params', str(frame_idx) + '.json')) as f:
        smplx_param = {k: torch.FloatTensor(v).cuda().view(-1) for k,v in json.load(f).items()}

    smplx_inputs_tuple = tuple(smplx_param[key] for key in wrapped_model.smplx_keys)
    cam_inputs_tuple = tuple(cam_param[key] for key in wrapped_model.cam_keys)
    dummy_inputs = smplx_inputs_tuple + cam_inputs_tuple

    input_names = wrapped_model.smplx_keys + wrapped_model.cam_keys
    # output_names = wrapped_model.output_names # 根據您在 Wrapper 中返回的內容命名

    with torch.no_grad():
        asset_2, asset_refined_2, offset, vert_neutral_pose, vert = wrapped_model(*dummy_inputs)
        for key in asset.keys():
            print((asset[key]-asset_2[key]).sum())
        for key in asset_refined.keys():
            print((asset_refined[key]-asset_refined_2[key]).sum())
    
    save_to_ply_3dgs_format(
        path='output_persona.ply',
        mean_3d=wrapped_model.mean_3d.cpu().numpy(),
        rgb=wrapped_model.rgb.cpu().numpy(),
        raw_opacity=wrapped_model.opacity.cpu().numpy(),
        raw_scale=wrapped_model.scale.cpu().numpy(),
        rotation=wrapped_model.rotation.cpu().numpy()
    )
    
    print(f"開始將模型轉換為 ONNX 格式，並儲存至 {args.output_path}...")
    sub_model_names=["refine","no_refine","static"]
    for sub_model_name in sub_model_names:
        wrapped_model = ModelWrapper(tester.model,smplx_param.keys(),cam_param.keys(),smplx_param, cam_param,sub_model_name).cuda().eval()
        output_names = wrapped_model.output_names 
        root, ext = os.path.splitext(args.output_path)
        new_path = f"{root}_{sub_model_name}{ext}"
        torch.onnx.export(
            wrapped_model,
            dummy_inputs,
            new_path,
            input_names=input_names,
            output_names=output_names,
            verbose=False,
            opset_version=16,
            export_params=True
        )
        
    print("模型轉換成功！")
    

    
        
    
if __name__ == "__main__":
    main()
