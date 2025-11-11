import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, OrthographicCameras, RasterizationSettings, MeshRasterizer, TexturesVertex

class MyChunkedGroupNorm(nn.Module):
    """
    一個自訂的 GroupNorm 層，增加了對大批次的自動切分處理功能。

    Attributes:
        chunk_size (int, optional): 
            單次處理的最大批次大小。如果輸入的批次大小超過此值，
            將會被自動切分成多個小塊進行處理。
            如果為 None，則不進行切分。預設為 None。
    """
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, chunk_size=None):
        super(MyChunkedGroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=affine)
        if chunk_size is None:
            chunk_size = 65535//num_groups
        self.chunk_size = chunk_size

    def _process_tensor(self, x):
        """
        核心處理邏輯：處理單個張量（可能是完整批次或一個小塊）。
        """
        # 檢查輸入張量的維度是否為 2D
        is_2d = (x.dim() == 2)
        
        # 如果是 2D 輸入 [N, C]，暫時擴展到 4D [N, C, 1, 1]
        if is_2d:
            reshaped_x = x.unsqueeze(-1).unsqueeze(-1)
        else:
            reshaped_x = x

        # 執行標準的 GroupNorm
        normed_x = self.gn(reshaped_x)

        # 如果原始輸入是 2D，則恢復其形狀 [N, C]
        if is_2d:
            return normed_x.squeeze(-1).squeeze(-1)
        else:
            return normed_x

    def forward(self, x):
        batch_size = x.shape[0]

        # 檢查是否需要切分
        if self.chunk_size is not None and batch_size > self.chunk_size:
            # --- 自動切分邏輯 ---
            # 1. 使用 torch.split 將輸入張量沿著批次維度切成多個小塊
            chunks = torch.split(x, self.chunk_size, dim=0)
            
            # 2. 對每個小塊獨立執行處理，並將結果收集起來
            processed_chunks = [self._process_tensor(chunk) for chunk in chunks]
            
            # 3. 使用 torch.cat 將處理後的小塊合併成一個完整的張量
            return torch.cat(processed_chunks, dim=0)
        else:
            # --- 標準處理邏輯 ---
            # 如果批次大小未超過閾值，或未設定 chunk_size，則直接處理
            return self._process_tensor(x)

    # 為了方便保存和載入模型，讓 state_dict 指向內部的 gn 層
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        self.gn._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def state_dict(self, *args, **kwargs):
        return self.gn.state_dict(*args, **kwargs)

def make_linear_layers(feat_dims, relu_final=True, use_gn=False):
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(nn.Linear(feat_dims[i], feat_dims[i+1]))

        # Do not use ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and relu_final):
            if use_gn:
                layers.append(MyChunkedGroupNorm(4, feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def rasterize(vert, face, cam_param, render_shape, bin_size=None):
    batch_size = vert.shape[0]
    face = torch.from_numpy(face).cuda()[None,:,:].repeat(batch_size,1,1)
    vert = torch.stack((-vert[:,:,0], -vert[:,:,1], vert[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(vert, face)

    cameras = PerspectiveCameras(focal_length=cam_param['focal'],
                                principal_point=cam_param['princpt'],
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(render_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=render_shape, blur_radius=0.0, faces_per_pixel=1, bin_size=bin_size)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    fragments = rasterizer(mesh)
    return fragments

def rasterize_to_uv(vertex_uv, face_uv, uvmap_shape):
    # scale UV coordinates to uvmap_shape
    vertex_uv = torch.stack((vertex_uv[:,:,0] * uvmap_shape[1], vertex_uv[:,:,1] * uvmap_shape[0]),2)
    vertex_uv = torch.cat((vertex_uv, torch.ones_like(vertex_uv[:,:,:1])),2) # add dummy depth
    vertex_uv = torch.stack((-vertex_uv[:,:,0], -vertex_uv[:,:,1], vertex_uv[:,:,2]),2) # reverse x- and y-axis following PyTorch3D axis direction
    mesh = Meshes(vertex_uv, face_uv)

    cameras = OrthographicCameras(
                                device='cuda',
                                in_ndc=False,
                                image_size=torch.LongTensor(uvmap_shape).cuda().view(1,2))
    raster_settings = RasterizationSettings(image_size=uvmap_shape, blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).cuda()
    outputs = rasterizer(mesh)
    return outputs

class MeshRenderer(nn.Module):
    def __init__(self):
        super(MeshRenderer, self).__init__()

    def forward(self, vert_mask, vert, face, cam_param, render_shape):
        batch_size = vert.shape[0]
        render_height, render_width = render_shape

        # rasterize
        vert = torch.bmm(cam_param['R'], vert.permute(0,2,1)).permute(0,2,1) + cam_param['t'].view(-1,1,3) # world coordinate -> camera coordinate
        fragments = rasterize(vert, face, cam_param, render_shape)

        # render texture
        vert_rgb = vert_mask[:,:,None].float()
        face = torch.LongTensor(face).cuda()
        render = TexturesVertex(vert_rgb).sample_textures(fragments, face)[0,:,:,0,:].permute(2,0,1) # 1, render_shape[0], render_shape[1]

        # fg mask
        pix_to_face = fragments.pix_to_face # batch_size, render_height, render_width, faces_per_pixel. invalid: -1
        pix_to_face_xy = pix_to_face[:,:,:,0] # Note: this is a packed representation
        is_fg = (pix_to_face_xy != -1).float()
        is_fg = is_fg[:,None,:,:]
        render = render * is_fg
        return render
