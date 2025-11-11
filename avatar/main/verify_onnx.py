import torch
import json
import argparse
import os.path as osp
import numpy as np
import onnxruntime # 載入 ONNX 執行環境
#python verify_onnx.py --subject_id gyeongsik --test_epoch 4 --motion_path /home/cgvmis418/ExAvatar_to_Unity/motions/jungkook_standing_next_to_you --onnx_path "../data/NeuMan/data/gyeongsik/human_model_ChunkedGroupNorm_lbs.onnx"

# 新增必要的 import，與 export_onnx.py 同步
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.ops import knn_points

# 假設您的 config, base, model, smpl_x 模組都在可導入的路徑中
from config import cfg
from base import Tester
from utils.smpl_x import smpl_x
from model import get_model
from plyfile import PlyData, PlyElement
# --- 新增輔助函式 ---
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
    features_rest = np.zeros((num_points, 45), dtype=np.float32)

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


# --- 步驟 1: 使用與 export_onnx.py 完全相同的 ModelWrapper ---
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        with torch.no_grad():
            mesh_neutral_pose, mesh_neutral_pose_wo_upsample, _, transform_mat_neutral_pose = model.module.human_gaussian.get_neutral_pose_human(jaw_zero_pose=True, use_id_info=True)
            joint_zero_pose = model.module.human_gaussian.get_zero_pose_human()

            # extract triplane feature
            tri_feat = model.module.human_gaussian.extract_tri_feature()
        
            # get Gaussian assets
            geo_feat = model.module.human_gaussian.geo_net(tri_feat)
            mean_offset = model.module.human_gaussian.mean_offset_net(geo_feat) # mean offset of Gaussians
            scale = model.module.human_gaussian.scale_net(geo_feat) # scale of Gaussians
            rgb = model.module.human_gaussian.rgb_net(tri_feat) # rgb of Gaussians
            mean_3d = mesh_neutral_pose + mean_offset # 大 pose
        # --- 核心修改：將傳入的常數張量註冊為 buffer ---
        self.register_buffer('tri_feat', tri_feat)
        self.register_buffer('scale', scale)
        self.register_buffer('rgb', rgb)
        self.register_buffer('mean_3d', mean_3d)
        self.register_buffer('joint_zero_pose', joint_zero_pose)
        self.register_buffer('mesh_neutral_pose_wo_upsample', mesh_neutral_pose_wo_upsample)
        self.register_buffer('transform_mat_neutral_pose', transform_mat_neutral_pose)
        self.register_buffer('parents', model.module.human_gaussian.smplx_layer.parents)
        self.register_buffer('skinning_weight', model.module.human_gaussian.skinning_weight)
        
        # 根據 smplx_params_smoothed_0.json 和 cam_params_0.json 的結構，定義輸入張量的鍵名和順序
        # **這個順序必須與後面建立 dummy_inputs 的順序完全一致**
        self.smplx_keys = [
             'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 
            'lhand_pose', 'rhand_pose', 'expr'
        ]
        # 注意: cam_params_0.json 中的 't' 在 module.py 中被當作 cam_param['t'] 使用，
        # 但 smplx_params_smoothed_0.json 中也有 'trans'。為避免混淆，請確認您的模型確實如此使用。
        # 根據 cam_params_0.json 的內容，這裡的鍵應為 'R', 't', 'focal', 'princpt'。
        self.cam_keys = []

    def forward(self, *inputs):
        # 將傳入的扁平化張量元組 (tuple of tensors) 重新組合成字典
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

        # 呼叫原始模型的 human_gaussian 部分
 
        # get pose-dependent Gaussian assets
        mean_offset_offset, scale_offset = self.model.module.human_gaussian.forward_geo_network(self.tri_feat, smplx_param)
        scale, scale_refined = torch.exp(self.scale).repeat(1,3), torch.exp(self.scale+scale_offset).repeat(1,3)
        mean_combined_offset, mean_offset_offset = self.model.module.human_gaussian.get_mean_offset_offset(smplx_param, mean_offset_offset)
        mean_3d_refined = self.mean_3d + mean_combined_offset # 大 pose

        # smplx facial expression offset
        smplx_expr_offset = (smplx_param['expr'][None,None,:] * self.model.module.human_gaussian.expr_dirs).sum(2)
        mean_3d = self.mean_3d + smplx_expr_offset # 大 pose
        mean_3d_refined = mean_3d_refined + smplx_expr_offset # 大 pose

        # get nearest vertex
        # for hands and face, assign original vertex index to use sknning weight of the original vertex
        nn_vertex_idxs = knn_points(mean_3d[None,:,:], self.mesh_neutral_pose_wo_upsample[None,:,:], K=1, return_nn=True).idx[0,:,0] # dimension: smpl_x.vertex_num_upsampled
        nn_vertex_idxs = self.model.module.human_gaussian.lr_idx_to_hr_idx(nn_vertex_idxs)
        mask = (self.model.module.human_gaussian.is_rhand + self.model.module.human_gaussian.is_lhand + self.model.module.human_gaussian.is_face) > 0
        updates = torch.arange(smpl_x.vertex_num_upsampled, device=nn_vertex_idxs.device, dtype=torch.int64)
        nn_vertex_idxs = torch.where(mask, updates, nn_vertex_idxs)

        # get transformation matrix of the nearest vertex and perform lbs
        # transform_mat_joint = self.model.module.human_gaussian.get_transform_mat_joint(self.transform_mat_neutral_pose, joint_zero_pose, smplx_param)
        # transform_mat_vertex = self.model.module.human_gaussian.get_transform_mat_vertex(transform_mat_joint, nn_vertex_idxs)
        # mean_3d = self.model.module.human_gaussian.lbs(mean_3d, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param
        # mean_3d_refined = self.model.module.human_gaussian.lbs(mean_3d_refined, transform_mat_vertex, smplx_param['trans']) # posed with smplx_param
        
        # forward to rgb network
        rgb = (torch.tanh(self.rgb) + 1) / 2
        
        rotation = matrix_to_quaternion(torch.eye(3).float().cuda()[None,:,:].repeat(smpl_x.vertex_num_upsampled,1,1)) # constant rotation
        opacity = torch.ones((smpl_x.vertex_num_upsampled,1)).float().cuda() # constant opacity
        # 根據 module.py 的定義，human_asset 是一個字典。
        # ONNX 導出需要返回一個張量或張量的元組，因此我們提取字典中的所有張量。
        return (
            mean_3d,
            opacity,
            scale,
            rotation, 
            rgb,
            mean_3d_refined,
            scale_refined,
            self.joint_zero_pose,
            self.transform_mat_neutral_pose,
            # nn_vertex_idxs,
            self.parents,
            self.skinning_weight[nn_vertex_idxs,:]
        )

def main():
    parser = argparse.ArgumentParser(description="Verify ONNX model against PyTorch model and export to PLY")
    parser.add_argument('--subject_id', type=str, required=True, help="Subject ID")
    parser.add_argument('--test_epoch', type=str, required=True, help="Model checkpoint epoch")
    parser.add_argument('--motion_path', type=str, required=True, help="Path to motion data")
    parser.add_argument('--onnx_path', type=str, default='human_gaussian_model.onnx', help="Path to the ONNX model to verify")
    args = parser.parse_args()

    cfg.set_args(args.subject_id)

    print("正在載入 PyTorch 模型並準備範例輸入...")
    tester = Tester(args.test_epoch)
    
    root_path = osp.join('..', 'data', cfg.dataset, 'data', cfg.subject_id)
    with open(osp.join(root_path, 'smplx_optimized', 'shape_param.json')) as f:
        shape_param = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'face_offset.json')) as f:
        face_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'joint_offset.json')) as f:
        joint_offset = torch.FloatTensor(json.load(f))
    with open(osp.join(root_path, 'smplx_optimized', 'locator_offset.json')) as f:
        locator_offset = torch.FloatTensor(json.load(f))
    smpl_x.set_id_info(shape_param, face_offset, joint_offset, locator_offset)
    
    tester.smplx_params = None
    tester._make_model()
    tester.model.eval()

    # *** 建議：使用與 export_onnx.py 相同的 frame_idx 以確保輸入完全一致 ***
    frame_idx = 100
    cam_param_file = osp.join(args.motion_path, 'cam_params', f'{frame_idx}.json')
    smplx_param_file = osp.join(args.motion_path, 'smplx_optimized', 'smplx_params_smoothed', f'{frame_idx}.json')

    with open(cam_param_file) as f:
        cam_param_dict = {k: torch.FloatTensor(v).cuda() for k, v in json.load(f).items()}
    with open(smplx_param_file) as f:
        smplx_param_dict = {k: torch.FloatTensor(v).cuda().view(-1) for k, v in json.load(f).items()}

    # --- 步驟 2: 更新模型初始化和輸入/輸出列表 ---
    wrapped_model = ModelWrapper(tester.model).cuda().eval()
    
    smplx_inputs_tuple = tuple(smplx_param_dict[key] for key in wrapped_model.smplx_keys)
    cam_inputs_tuple = tuple(cam_param_dict[key] for key in wrapped_model.cam_keys)
    dummy_inputs = smplx_inputs_tuple + cam_inputs_tuple
    
    input_names = wrapped_model.smplx_keys + wrapped_model.cam_keys
    # 更新輸出的名稱列表以匹配 ModelWrapper 的回傳值
    output_names = [
        'mean_3d',
            'opacity',
            'scale',
            'rotation', 
            'rgb',
            'mean_3d_refined',
            'scale_refined',
            'joint_zero_pose',
            'transform_mat_neutral_pose',
            'parents',
            'skinning_weight'
    ]
    print("PyTorch 模型與輸入準備完成。")

    # --- 步驟 3: 執行 PyTorch 模型推論 ---
    print("\n正在執行 PyTorch 模型推論...")
    with torch.no_grad():
        pytorch_outputs = wrapped_model(*dummy_inputs)
    pytorch_outputs_np = [t.cpu().numpy() for t in pytorch_outputs]
    print("PyTorch 推論完成。")

    # --- 步驟 4: 載入 ONNX 模型並執行推論 ---
    print(f"\n正在載入 ONNX 模型 '{args.onnx_path}' 並執行推論...")
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    ort_inputs = {
        input_name: input_tensor.cpu().numpy()
        for input_name, input_tensor in zip(input_names, dummy_inputs)
    }
    onnx_outputs = ort_session.run(output_names, ort_inputs)
    print("ONNX 推論完成。")

    # --- 步驟 5: 比較兩個模型的輸出 ---
    print("\n--- 輸出結果比較 ---")
    TOLERANCE = 1e-4
    all_match = True

    for i in range(len(output_names)):
        pytorch_res = pytorch_outputs_np[i]
        onnx_res = onnx_outputs[i]
        output_name = output_names[i]

        print(f"\n--- 輸出 '{output_name}' ---")
        if pytorch_res.shape != onnx_res.shape:
            print(f"   狀態: ❌ 形狀不匹配!")
            print(f"   PyTorch shape: {pytorch_res.shape}")
            print(f"   ONNX shape:    {onnx_res.shape}")
            all_match = False
            continue

        abs_diff = np.abs(pytorch_res - onnx_res)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        num_elements = pytorch_res.size
        outlier_count = np.sum(abs_diff > TOLERANCE)
        error_ratio = outlier_count / num_elements
        
        is_close = outlier_count == 0

        if is_close:
            status_icon = "✅"
            status_text = "驗證通過"
            all_match = all_match and True
        else:
            status_icon = "❌"
            status_text = "驗證失敗"
            all_match = False
        
        print(f"   狀態: {status_icon} {status_text}")
        print(f"   最大絕對誤差: {max_diff:.6g}")
        print(f"   平均絕對誤差 (MAE): {mean_diff:.6g}")
        print(f"   誤差 > {TOLERANCE} 的元素數量: {outlier_count} / {num_elements}")
        print(f"   出錯比例: {error_ratio:.4%}")
    
    print("\n\n--- 驗證總結 ---")
    if all_match:
        print("🎉 所有輸出均在容忍度內！ONNX 模型已成功驗證。")
    else:
        print("💔 發現部分輸出不匹配。請根據上述詳細指標進行評估。")

    # --- NEW CODE ---
    # 在驗證結束後，將 PyTorch 和 ONNX 的輸出儲存為 .ply 檔案
    print("\n\n--- 儲存 3DGS PLY 檔案 ---")

    # 儲存 PyTorch 模型的輸出
    save_to_ply_3dgs_format(
        path='output_pytorch.ply',
        mean_3d=pytorch_outputs_np[output_names.index('mean_3d')],
        rgb=pytorch_outputs_np[output_names.index('rgb')],
        raw_opacity=pytorch_outputs_np[output_names.index('opacity')],
        raw_scale=pytorch_outputs_np[output_names.index('scale')],
        rotation=pytorch_outputs_np[output_names.index('rotation')]
    )
    
    # 儲存 ONNX 模型的輸出
    save_to_ply_3dgs_format(
        path='output_onnx.ply',
        mean_3d=onnx_outputs[output_names.index('mean_3d')],
        rgb=onnx_outputs[output_names.index('rgb')],
        raw_opacity=onnx_outputs[output_names.index('opacity')],
        raw_scale=onnx_outputs[output_names.index('scale')],
        rotation=onnx_outputs[output_names.index('rotation')]
    )


if __name__ == "__main__":
    main()