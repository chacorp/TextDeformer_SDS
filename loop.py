import clip
import kornia
import os
import pathlib
import pymeshlab
import shutil
import torch
import torchvision
import logging
import yaml

import numpy as np
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt

from easydict import EasyDict

from NeuralJacobianFields import SourceMesh

from nvdiffmodeling.src     import obj
from nvdiffmodeling.src     import util
from nvdiffmodeling.src     import mesh
from nvdiffmodeling.src     import render
from nvdiffmodeling.src     import texture
from nvdiffmodeling.src     import regularizer

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utilities.video import Video
from utilities.helpers import cosine_avg, create_scene, get_vp_map
from utilities.camera import CameraBatch, get_camera_params
from utilities.clip_spatial import CLIPVisualEncoder
from utilities.resize_right import resize, cubic, linear, lanczos2, lanczos3


from torchvision.transforms import ToPILImage
from stable_diffusion import StableDiffusion

def init_diffusion(device):
    MODEL_NAME = '/source/kseo/huggingface_cache/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819'
    CACHE_DIR = "/source/kseo/huggingface_cache"
    diffusion_model = StableDiffusion(
            device, 
            model_name   = MODEL_NAME,
            cache_dir    = CACHE_DIR,
            latent_mode  = True,
        )

    for p in diffusion_model.parameters():
        p.requires_grad = False

    return diffusion_model

def get_view_direction(elev, azim, top=50, front=90):
    two_pi = 2*np.pi
    angle = np.deg2rad(45) % two_pi
    
    top   = np.deg2rad(top) % two_pi
    front = np.deg2rad(front) % two_pi
    azim = azim % two_pi
    elev = elev % two_pi
    
    view = torch.zeros(elev.shape[0], dtype=torch.long)
    
    # first determine by azim
    view[   (front-angle <= azim) & (azim < front+angle)   ] = 0  # front 
    view[  (two_pi-angle <= azim) | (azim < front-angle)   ] = 1  # left side
    view[ (2*front+angle <= azim) & (azim < two_pi-angle)  ] = 2  # back
    view[   (front * 1.5 <= azim) & (azim < 2*front+angle) ] = 3  # right side

#     if (front-angle <= azim)       & (azim < front+angle):
#         res = res * 0  # front 
#         print(front-angle, azim, front+angle)
# #     elif (two_pi-angle <= two_pi-azim) & (azim < front-angle):
#     elif (two_pi-angle <= azim) | (azim < front-angle):
#         res = res * 0 + 1  # left side
#         print(two_pi-angle, azim, front-angle)
#     elif (2*front+angle <= azim)       & (azim < two_pi-angle):
#         res = res * 0 + 2  # back
#         print(2*front+angle, azim, two_pi-angle)
#     elif (front * 1.5 <= azim)       & (azim < 2*front+angle):
#         res = res * 0 + 3  # right side
#         print(front * 1.5, azim, 2*front+angle)
#     else:
#         print('no')
        
    view[elev > top] = 4                                          # overhead
    view[(two_pi-elev) < (two_pi-top)] = 5                        # bottom
    return view
#     return ['front', 'left side', 'back', 'right side', 'overhead', 'bottom'][res[0]]


def loop(cfg):
    output_path = pathlib.Path(cfg['output_path'])
    os.makedirs(output_path, exist_ok=True)
    with open(output_path / 'config.yml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    cfg = EasyDict(cfg)
    
    
    ### Configure ------------------------------------------------------------------------
    print(f'Output directory {cfg.output_path} created')

    device = torch.device(f'cuda:{cfg.gpu}')
    torch.cuda.set_device(device)

    
    ### Load stable-diffusion ------------------------------------------------------------
    diffusion          = init_diffusion(device)
    
    
    text_z_list = []
#     text = []
#     text_neg = []
    for d in ['front', 'left side', 'back', 'right side', 'overhead', 'bottom']:
#         text.append(f"{cfg.text_prompt}, {d} view")
#         text_neg.append(f"{cfg.base_text_prompt}, {d} view")
        text     = [f"{cfg.text_prompt}, {d} view"]
        text_neg = [f"{cfg.base_text_prompt}, {d} view"]
        text_z_list.append(diffusion.get_text_embeds(prompt=text, negative_prompt=text_neg))
        
    text_z_list = torch.stack(text_z_list)
#     text_z_list = diffusion.get_text_embeds(prompt=text, negative_prompt=text_neg)
    
    """
    [v] get results from no deepfeat
        - seems to learn well
    [v] prompt to embedding with view directions
        - for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']
    [ ] get view direction per each camera
    
    """
    
    
    ### Load CLIP ------------------------------------------------------------------------
    print('Loading CLIP Models')
    model, _ = clip.load(cfg.clip_model, device=device)
    fe = CLIPVisualEncoder(cfg.consistency_clip_model, cfg.consistency_vit_stride, device)

    clip_mean = torch.tensor([0.48154660, 0.45782750, 0.40821073], device=device)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)

    
    ### Rendering ------------------------------------------------------------------------
    # output video
    video = Video(cfg.output_path)

    # GL Context
#     glctx = dr.RasterizeGLContext()
    glctx = dr.RasterizeCudaContext()

    print(f'Target text prompt is {cfg.text_prompt}')
    print(f'Base text prompt is {cfg.base_text_prompt}')
    with torch.no_grad():
        text_embeds = clip.tokenize(cfg.text_prompt).to(device)
        base_text_embeds = clip.tokenize(cfg.base_text_prompt).to(device)
        text_embeds = model.encode_text(text_embeds).detach()
        target_text_embeds = text_embeds.clone() / text_embeds.norm(dim=1, keepdim=True)

        delta_text_embeds = text_embeds - model.encode_text(base_text_embeds)
        delta_text_embeds = delta_text_embeds / delta_text_embeds.norm(dim=1, keepdim=True)

        
    ### Mesh -----------------------------------------------------------------------------
    os.makedirs(output_path / 'tmp', exist_ok=True)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(cfg.mesh)

    if cfg.retriangulate:
        print('Retriangulating shape')
        ms.meshing_isotropic_explicit_remeshing()
    
    if not ms.current_mesh().has_wedge_tex_coord():
        # some arbitrarily high number
        ms.compute_texcoord_parametrization_triangle_trivial_per_wedge(textdim=10000)
    
    ms.save_current_mesh(str(output_path / 'tmp' / 'mesh.obj'))

    load_mesh = obj.load_obj(str(output_path / 'tmp' / 'mesh.obj'))
    load_mesh = mesh.unit_size(load_mesh)
#     import pdb;pdb.set_trace() # load_mesh.v_pos.min() load_mesh.v_pos.max()
#     load_mesh.v_pos = (load_mesh.v_pos + 1) * 0.5
    
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=load_mesh.v_pos.cpu().numpy(), face_matrix=load_mesh.t_pos_idx.cpu().numpy()))
    ms.save_current_mesh(str(output_path / 'tmp' / 'mesh.obj'), save_vertex_color=False)

    ### Texture --------------------------------------------------------------------------
    # TODO: Need these for rendering even if we don't optimize textures
    texture_map = texture.create_trainable(np.random.uniform(size=[512]*2 + [3], low=0.0, high=1.0), [512]*2, True)
    normal_map = texture.create_trainable(np.array([0, 0, 1]), [512]*2, True)
    specular_map = texture.create_trainable(np.array([0, 0, 0]), [512]*2, True)

    load_mesh = mesh.Mesh(
        material={
            'bsdf': cfg.bsdf,
            'kd': texture_map,
            'ks': specular_map,
            'normal': normal_map,
        },
        base=load_mesh # Get UVs from original loaded mesh
    )
    
    ### Jacobian -------------------------------------------------------------------------
    jacobian_source = SourceMesh.SourceMesh(0, str(output_path / 'tmp' / 'mesh.obj'), {}, 1, ttype=torch.float)
    
    if len(list((output_path / 'tmp').glob('*.npz'))) > 0:
        logging.warn(f'Using existing Jacobian .npz files in {str(output_path)}/tmp/ ! Please check if this is intentional.')
    jacobian_source.load()
    jacobian_source.to(device)

    with torch.no_grad():
        gt_jacobians = jacobian_source.jacobians_from_vertices(load_mesh.v_pos.unsqueeze(0))
    gt_jacobians.requires_grad_(True)

    ## v[f].shape == gt_jacobians
    optimizer = torch.optim.Adam([gt_jacobians], lr=cfg.lr)
    
    ### Camera set -----------------------------------------------------------------------
    cams_data = CameraBatch(
        cfg.train_res,
        [cfg.dist_min, cfg.dist_max],
        [cfg.azim_min, cfg.azim_max],
        [cfg.elev_alpha, cfg.elev_beta, cfg.elev_max],
        [cfg.fov_min, cfg.fov_max],
        cfg.aug_loc,
        cfg.aug_light,
        cfg.aug_bkg,
        cfg.batch_size,
        rand_solid=True
    )
    cams = torch.utils.data.DataLoader(cams_data, cfg.batch_size, num_workers=0, pin_memory=True)
    
    ### Loggings -------------------------------------------------------------------------
    best_losses = {'CLIP': np.inf, 'total': np.inf}

    for out_type in ['final', 'best_clip', 'best_total']:
        os.makedirs(output_path / f'mesh_{out_type}', exist_ok=True)
    os.makedirs(output_path / 'images', exist_ok=True)
    logger = SummaryWriter(str(output_path / 'logs'))

    rot_ang = 0.0
    t_loop = tqdm(range(cfg.epochs), leave=False)

    if cfg.resize_method == 'cubic':
        resize_method = cubic
    elif cfg.resize_method == 'linear':
        resize_method = linear
    elif cfg.resize_method == 'lanczos2':
        resize_method = lanczos2
    elif cfg.resize_method == 'lanczos3':
        resize_method = lanczos3

    for it in t_loop:
#         import pdb;pdb.set_trace()
        
        # updated vertices from jacobians
        n_vert = jacobian_source.vertices_from_jacobians(gt_jacobians).squeeze()

        # TODO: More texture code required to make it work ...
        ready_texture = texture.Texture2D(
            kornia.filters.gaussian_blur2d(
                load_mesh.material['kd'].data.permute(0, 3, 1, 2),
                kernel_size=(7, 7),
                sigma=(3, 3),
            ).permute(0, 2, 3, 1).contiguous()
        )
        ### using gray texture
        kd_notex = texture.Texture2D(torch.full_like(ready_texture.data, 0.5))

        ready_specular = texture.Texture2D(
            kornia.filters.gaussian_blur2d(
                load_mesh.material['ks'].data.permute(0, 3, 1, 2),
                kernel_size=(7, 7),
                sigma=(3, 3),
            ).permute(0, 2, 3, 1).contiguous()
        )

        ready_normal = texture.Texture2D(
            kornia.filters.gaussian_blur2d(
                load_mesh.material['normal'].data.permute(0, 3, 1, 2),
                kernel_size=(7, 7),
                sigma=(3, 3),
            ).permute(0, 2, 3, 1).contiguous()
        )
            
        # Final mesh
        m = mesh.Mesh(
            n_vert,
            load_mesh.t_pos_idx,
            material={
                'bsdf': cfg.bsdf,
                'kd': kd_notex,
                'ks': ready_specular,
                'normal': ready_normal,
            },
            base=load_mesh # gets uvs etc from here
        )

        render_mesh = create_scene([m.eval()], sz=512)
        if it == 0:
            base_mesh = render_mesh.clone()
            base_mesh = mesh.auto_normals(base_mesh)
            base_mesh = mesh.compute_tangents(base_mesh)
        render_mesh = mesh.auto_normals(render_mesh)
        render_mesh = mesh.compute_tangents(render_mesh)
        
        # Logging mesh -----------------------------------------------------------------------
        log_mesh = it % cfg.log_interval == 0
        if log_mesh:
            with torch.no_grad():
                params = get_camera_params(
                    cfg.log_elev,
                    rot_ang,
                    cfg.log_dist,
                    cfg.log_res,
                    cfg.log_fov,
                )
#                 rot_ang += 1
                log_mesh = mesh.unit_size(render_mesh.eval(params))
                log_image = render.render_mesh(
                    glctx,
                    log_mesh,
                    params['mvp'],
                    params['campos'],
                    params['lightpos'],
                    cfg.log_light_power,
                    cfg.log_res,
                    1,
                    background=torch.ones(1, cfg.log_res, cfg.log_res, 3).to(device)
                )

                log_image = video.ready_image(log_image)
                logger.add_mesh('predicted_mesh', vertices=log_mesh.v_pos.unsqueeze(0), faces=log_mesh.t_pos_idx.unsqueeze(0), global_step=it)
        
        if cfg.adapt_dist and it > 0:
            with torch.no_grad():
                v_pos = m.v_pos.clone()
                vmin = v_pos.amin(dim=0)
                vmax = v_pos.amax(dim=0)
                v_pos -= (vmin + vmax) / 2
                mult = torch.cat([v_pos.amin(dim=0), v_pos.amax(dim=0)]).abs().amax().cpu()
                cams.dataset.dist_min = cfg.dist_min * mult
                cams.dataset.dist_max = cfg.dist_max * mult

        ### Render ---------------------------------------------------------------------------
        params_camera = next(iter(cams))
        for key in params_camera:
            params_camera[key] = params_camera[key].to(device)
        
        final_mesh = render_mesh.eval(params_camera)
        
        train_render = render.render_mesh(
            glctx,
            final_mesh,
            params_camera['mvp'],
            params_camera['campos'],
            params_camera['lightpos'],
            cfg.light_power,
            cfg.train_res,
            spp=1,
            num_layers=1,
            msaa=False,
            background=params_camera['bkgs']
        ).permute(0, 3, 1, 2)
        
        ######################################################################################
        ## experiment
        """
        TODO:
        1. load stable-diffusion [v]
        2. SDS [v]
        """
        optimizer.zero_grad()
        
        dirs = get_view_direction(params_camera['elev'], params_camera['azim'])
        
        text_z = text_z_list[dirs]
        
        ### RGB image -> latent
        # train_render = resize(train_render, out_shape=(224, 224), interp_method=resize_method) # [25, 3, 512, 512]
        train_render = resize(train_render, out_shape=(64, 64), interp_method=resize_method) * 2.0 - 1.0
        est_latents = torch.einsum('ib,abcd->aicd', diffusion.linear_rgb_estimator, train_render)
#         latent = diffusion.encode_imgs(train_render[0][None])
#         loss_guidance = diffusion.train_step(text_z[0], est_latents[0][None])
#         import pdb;pdb.set_trace()
    
        ## even: uncond, odd:cond
        # torch.stack(torch.arange(12).chunk(6))
        # tensor([[ 0,  1],
        #         [ 2,  3],
        #         [ 4,  5],
        #         [ 6,  7],
        #         [ 8,  9],
        #         [10, 11]])    
        # torch.stack(torch.arange(12).chunk(6)).permute(1,0).reshape(-1)
        # tensor([ 0,  2,  4,  6,  8, 10,  1,  3,  5,  7,  9, 11])
        loss_guidance = diffusion.train_step(text_z.permute(1,0,2,3).reshape(-1,77,768), est_latents)
        ######################################################################################
        
        
#         train_rast_map = render.render_mesh(
#             glctx,
#             final_mesh,
#             params_camera['mvp'],
#             params_camera['campos'],
#             params_camera['lightpos'],
#             cfg.light_power,
#             cfg.train_res,
#             spp=1,
#             num_layers=1,
#             msaa=False,
#             background=params_camera['bkgs'],
#             return_rast_map=True
#         )

#         if it == 0:
#             params_camera = next(iter(cams))
#             for key in params_camera:
#                 params_camera[key] = params_camera[key].to(device)
#         base_render = render.render_mesh(
#             glctx,
#             base_mesh.eval(params_camera),
#             params_camera['mvp'],
#             params_camera['campos'],
#             params_camera['lightpos'],
#             cfg.light_power,
#             cfg.train_res,
#             spp=1,
#             num_layers=1,
#             msaa=False,
#             background=params_camera['bkgs'],
#         ).permute(0, 3, 1, 2)
#         base_render = resize(base_render, out_shape=(224, 224), interp_method=resize_method)
        
        if it % cfg.log_interval_im == 0:
            log_idx = torch.randperm(cfg.batch_size)[:5]
            s_log = train_render[log_idx, :, :, :]
            s_log = torchvision.utils.make_grid(s_log)
            ndarr = s_log.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(ndarr)
            im.save(str(output_path / 'images' / 'epoch_{:04}.png'.format(it) ))

            obj.write_obj(
                str(output_path / 'mesh_final'),
                m.eval()
            )
        
#         optimizer.zero_grad()

#         import pdb;pdb.set_trace()
        # ToPILImage()(train_render[0]).save('test.png')

        
#         loss_guidance = diffusion.train_step(text_z, train_render)

        ### Loss -----------------------------------------------------------------------------
    
#         # CLIP similarity losses
#         normalized_clip_render = (train_render - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]
#         image_embeds = model.encode_image(
#             normalized_clip_render
#         )
#         with torch.no_grad():
#             normalized_base_render = (base_render - clip_mean[None, :, None, None]) / clip_std[None, :, None, None]
#             base_embeds = model.encode_image(normalized_base_render)
        
#         orig_image_embeds  = image_embeds.clone() / image_embeds.norm(dim=1, keepdim=True)
#         delta_image_embeds = image_embeds - base_embeds
#         delta_image_embeds = delta_image_embeds / delta_image_embeds.norm(dim=1, keepdim=True)

#         clip_loss = cosine_avg(orig_image_embeds, target_text_embeds)
#         delta_clip_loss = cosine_avg(delta_image_embeds, delta_text_embeds)
#         logger.add_scalar('clip_loss', clip_loss, global_step=it)
#         logger.add_scalar('delta_clip_loss', delta_clip_loss, global_step=it)

#         # Jacobian regularization
#         r_loss = (((gt_jacobians) - torch.eye(3, 3, device=device)) ** 2).mean()
#         logger.add_scalar('jacobian_regularization', r_loss, global_step=it)

#        # Consistency loss
#        # Get mapping from vertex to pixels
#         curr_vp_map = get_vp_map(final_mesh.v_pos, params_camera['mvp'], 224)
#         for idx, rast_faces in enumerate(train_rast_map[:, :, :, 3].view(cfg.batch_size, -1)):
#             u_faces = rast_faces.unique().long()[1:] - 1
#             t = torch.arange(len(final_mesh.v_pos), device=device)
#             u_ret = torch.cat([t, final_mesh.t_pos_idx[u_faces].flatten()]).unique(return_counts=True)
#             non_verts = u_ret[0][u_ret[1] < 2]
#             curr_vp_map[idx][non_verts] = torch.tensor([224, 224], device=device)
        
#         # Get mapping from vertex to patch
#         med = (fe.old_stride - 1) / 2
#         curr_vp_map[curr_vp_map < med] = med
#         curr_vp_map[(curr_vp_map > 224 - fe.old_stride) & (curr_vp_map < 224)] = 223 - med
#         curr_patch_map = ((curr_vp_map - med) / fe.new_stride).round()
#         flat_patch_map = curr_patch_map[..., 0] * (((224 - fe.old_stride) / fe.new_stride) + 1) + curr_patch_map[..., 1]
        
#        # Deep features
#         patch_feats = fe(normalized_clip_render)
#         flat_patch_map[flat_patch_map > patch_feats[0].shape[-1] - 1] = patch_feats[0].shape[-1]
#         flat_patch_map = flat_patch_map.long()[:, None, :].repeat(1, patch_feats[0].shape[1], 1)

#         deep_feats = patch_feats[cfg.consistency_vit_layer]
#         deep_feats = torch.nn.functional.pad(deep_feats, (0, 1))
#         deep_feats = torch.gather(deep_feats, dim=2, index=flat_patch_map)
#         deep_feats = torch.nn.functional.normalize(deep_feats, dim=1, eps=1e-6)

#         elev_d = torch.cdist(params_camera['elev'].unsqueeze(1), params_camera['elev'].unsqueeze(1)).abs() < torch.deg2rad(torch.tensor(cfg.consistency_elev_filter))
#         azim_d = torch.cdist(params_camera['azim'].unsqueeze(1), params_camera['azim'].unsqueeze(1)).abs() < torch.deg2rad(torch.tensor(cfg.consistency_azim_filter))

#         cosines = torch.einsum('ijk, lkj -> ilk', deep_feats, deep_feats.permute(0, 2, 1))
#         cosines = (cosines * azim_d.unsqueeze(-1) * elev_d.unsqueeze(-1)).permute(2, 0, 1).triu(1)
#         consistency_loss = cosines[cosines != 0].mean()
#         logger.add_scalar('consistency_loss', consistency_loss, global_step=it)

        ### Total Loss -----------------------------------------------------------------------
#         total_loss = cfg.clip_weight * clip_loss + cfg.delta_clip_weight * delta_clip_loss + \
#             cfg.regularize_jacobians_weight * r_loss - cfg.consistency_loss_weight * consistency_loss

#         total_loss = cfg.clip_weight * clip_loss + cfg.delta_clip_weight * delta_clip_loss + cfg.regularize_jacobians_weight * r_loss
    
#         logger.add_scalar('total_loss', total_loss, global_step=it)

#         if best_losses['total'] > total_loss and log_mesh:
#             best_losses['total'] = total_loss.detach()
#             obj.write_obj(
#                 str(output_path / 'mesh_best_total'),
#                 m.eval()
#             )
#         if best_losses['CLIP'] > clip_loss:
#             best_losses['CLIP'] = clip_loss.detach()
#             obj.write_obj(
#                 str(output_path / 'mesh_best_clip'),
#                 m.eval()
#             )

#         total_loss.backward()
        optimizer.step()
#         t_loop.set_description(f'CLIP Loss = {clip_loss.item()}, Total Loss = {total_loss.item()}')
    
    video.close()
    obj.write_obj(
        str(output_path / 'mesh_final'),
        m.eval()
    )
    
    return
