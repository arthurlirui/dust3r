import sys
import os.path
import cv2

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_cpu, collate_with_cat

import tqdm
import torch
import json
import numpy as np
from pyquaternion import Quaternion


def split_img(img_path):
    img = cv2.imread(img_path)
    img0 = img[:, :1920, :]
    img1 = img[:, 1920:, :]
    return img0, img1


def render_depth():
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    #model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    model_name = os.path.join('./checkpoints/', 'DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    img_path = 'D:\\Data\\1cm\\test7'
    img_list = os.listdir(img_path)
    # img_list = ['IMG_0007.jpg', 'IMG_0008.jpg', 'IMG_0009.jpg', 'IMG_0010.jpg']
    # img_list = ['IMG_0011.jpg', 'IMG_0012.jpg', 'IMG_0013.jpg', 'IMG_0014.jpg']
    img_full_list = [os.path.join(img_path, f) for f in img_list]
    images = load_images(img_full_list, size=640, verbose=True, dual_camera=False)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    #output = inference(pairs, model, device, batch_size=batch_size)
    result = []

    # first, check if all images have the same size
    #multiple_shapes = not (check_if_same_size(pairs))
    #if multiple_shapes:  # force bs=1
    #    batch_size = 1
    batch_size = 1
    for i in tqdm.trange(0, len(pairs), batch_size, disable=False):
        #res = loss_of_one_batch(collate_with_cat(pairs[i:i + batch_size]), model, None, device)
        view1, view2 = collate_with_cat(pairs[i:i+batch_size])
        #result.append(to_cpu(res))
        with torch.cuda.amp.autocast(enabled=True):
            view1['img'] = view1['img'].to(device, non_blocking=True)
            view2['img'] = view2['img'].to(device, non_blocking=True)
            #view1, view2 = make_batch_symmetric(batch)
            pred1, pred2 = model(view1, view2)

    #result = collate_with_cat(result, lists=multiple_shapes)
    return pred1, pred2


def load_camera_files(json_path):
    res = {}
    #for key in img_dict:
    #    res[key] = {'device_id': key, 'intrinsic': None, 'extrinsic': None}

    with open(json_path, 'r', encoding='utf-8') as f:
        res = json.load(f)
        for key in res:
            res[key]['intrinsic'] = np.array(res[key]['intrinsic'])
            res[key]['extrinsic'] = np.array(res[key]['extrinsic'])
            q = Quaternion(matrix=res[key]['extrinsic'])
            res[key]['Q'] = np.array([q[0], q[1], q[2], q[3]])
            #res[key]['rvec'] = np.array(res[key]['rvec'])
            #res[key]['tvec'] = np.array(res[key]['tvec'])
    return res


def main_run():
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # model_name = os.path.join('./checkpoint/', 'DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth')
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    #img_path = 'D:\\Data\\ob\\0607\\test\\rgb'
    img_path = 'D:\\Data\\ob\\0607\\test\\rgb'
    img_list = os.listdir(img_path)
    # img_list = ['IMG_0007.jpg', 'IMG_0008.jpg', 'IMG_0009.jpg', 'IMG_0010.jpg']
    # img_list = ['IMG_0011.jpg', 'IMG_0012.jpg', 'IMG_0013.jpg', 'IMG_0014.jpg']
    img_full_list = sorted([os.path.join(img_path, f) for f in img_list])
    images = load_images(img_full_list, size=640, verbose=True, dual_camera=False)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    json_path = 'D:\\Data\\ob\\0607\\test'
    cameras = load_camera_files(json_path=os.path.join(json_path, 'camera_info.json'))
    keys = [os.path.basename(f).split('_')[0] for f in img_full_list]

    # QW QX QY QZ X Y Z
    known_poses = [cameras[key]['Q'] for key in keys]

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.preset_pose(known_poses=None)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)
    #loss = scene.compute_global_alignment(init="known_poses", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # visualize reconstruction
    scene.show()

    # find 2D-2D matches between the two images
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f'found {num_matches} matches')
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # visualize a few matches
    import numpy as np
    from matplotlib import pyplot as pl
    n_viz = 10
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)


if __name__ == '__main__':
    if False:
        path = 'D:\\Data\\ob\\0607\\tmp'
        path = os.path.join(path, 'camera_info.json')
        res = load_camera_files(json_path=path)
        from pprint import pprint
        pprint(res)

    if True:
        main_run()

    if False:
        pred1, pred2 = render_depth()

        # visualize depth
        cv2.imshow('pred1', pred1)
        cv2.imshow('pred2', pred2)


