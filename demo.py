# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This work has been modified by Rerun. Additions that belong to Rerun are
# also released under the same license as the original work.

import os
import numpy as np
import cv2
from tqdm import tqdm

import torch
from pytorch3d.io.obj_io import load_obj

import main_mcc
import mcc_model
import util.misc as misc
import rerun as rr
from typing import Final
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.modeling import Sam
import requests
from pathlib import Path
from engine_mcc import prepare_data

MODEL_DIR: Final = Path(os.path.dirname(__file__)) / "checkpoint"
MODEL_URLS: Final = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "mcc": "https://dl.fbaipublicfiles.com/MCC/co3dv2_all_categories.pth"
}

def download_with_progress(url: str, dest: Path) -> None:
    """Download file with tqdm progress bar."""
    chunk_size = 1024 * 1024
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as dest_file:
        with tqdm(
            desc="Downloading model", total=total_size, unit="iB", unit_scale=True, unit_divisor=1024
        ) as progress:
            for data in resp.iter_content(chunk_size):
                dest_file.write(data)
                progress.update(len(data))


def get_downloaded_model_path(model_name: str) -> Path:
    """Fetch the segment-anything model to a local cache directory."""
    model_url = MODEL_URLS[model_name]

    model_location = MODEL_DIR / model_url.split("/")[-1]
    if not model_location.exists():
        os.makedirs(MODEL_DIR, exist_ok=True)
        download_with_progress(model_url, model_location)

    return model_location


def create_sam(model: str, device: str) -> Sam:
    """Load the segment-anything model, fetching the model-file as necessary."""
    model_path = get_downloaded_model_path(model)

    sam = sam_model_registry[model](checkpoint=model_path)
    return sam.to(device=device)


def log_points(occupancy, xyz, color, threshold=0.1):
    """Given occupancy, xyz, and color, log points using rerun. """
    occupancy = torch.nn.Sigmoid()(occupancy)
    pos = occupancy > threshold
    points = xyz[pos].reshape((-1, 3))
    features = color[None][pos].reshape((-1, 3))
    good_points = points[:, 0] != -100

    if good_points.sum() == 0:
        return

    rr.log(
        "predicted_points",
        rr.Points3D(
            positions=points[good_points].cpu(),
            colors=features[good_points].numpy(force=True)
        )
    )


def run_viz(model, samples, device, args):
    model.eval()

    seen_xyz, valid_seen_xyz, unseen_xyz, unseen_rgb, labels, seen_images = prepare_data(
        samples, device, is_train=False, args=args, is_viz=True
    )
    pred_occupy = []
    pred_colors = []

    max_n_unseen_fwd = 2000

    model.cached_enc_feat = None
    num_passes = int(np.ceil(unseen_xyz.shape[1] / max_n_unseen_fwd))
    for p_idx in tqdm(range(num_passes)):
        rr.set_time_sequence("num_passes", p_idx+1)
        p_start = p_idx     * max_n_unseen_fwd
        p_end = (p_idx + 1) * max_n_unseen_fwd
        cur_unseen_xyz = unseen_xyz[:, p_start:p_end]
        cur_unseen_rgb = unseen_rgb[:, p_start:p_end].zero_()
        cur_labels = labels[:, p_start:p_end].zero_()

        with torch.no_grad():
            _, pred = model(
                seen_images=seen_images,
                seen_xyz=seen_xyz,
                unseen_xyz=cur_unseen_xyz,
                unseen_rgb=cur_unseen_rgb,
                unseen_occupy=cur_labels,
                cache_enc=True,
                valid_seen_xyz=valid_seen_xyz,
            )
        pred_occupy.append(pred[..., 0].cpu())
        if args.regress_color:
            pred_colors.append(pred[..., 1:].reshape((-1, 3)))
        else:
            pred_colors.append(
                (
                    torch.nn.Softmax(dim=2)(
                        pred[..., 1:].reshape((-1, 3, 256)) / args.temperature
                    ) * torch.linspace(0, 1, 256, device=pred.device)
                ).sum(axis=2)
            )

        if p_idx != 0:
            vis_occupy = torch.cat(pred_occupy, dim=1)
            vis_colors = torch.cat(pred_colors, dim=0)
            vis_xyz = unseen_xyz[:, :p_end, :]
            log_points(
                vis_occupy,
                vis_xyz,
                vis_colors)


def pad_image(im, value):
    if im.shape[0] > im.shape[1]:
        diff = im.shape[0] - im.shape[1]
        return torch.cat([im, (torch.zeros((im.shape[0], diff, im.shape[2])) + value)], dim=1)
    else:
        diff = im.shape[1] - im.shape[0]
        return torch.cat([im, (torch.zeros((diff, im.shape[1], im.shape[2])) + value)], dim=0)


def normalize(seen_xyz):
    seen_xyz = seen_xyz / (seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].var(dim=0) ** 0.5).mean()
    seen_xyz = seen_xyz - seen_xyz[torch.isfinite(seen_xyz.sum(dim=-1))].mean(axis=0)
    return seen_xyz


def get_intrinsics(H,W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])


def backproject_depth_to_pointcloud(depth, rotation=np.eye(3), translation=np.zeros(3)):
    intrinsics = get_intrinsics(depth.shape[0], depth.shape[1])
    # Get the depth map shape
    height, width = depth.shape

    # Create a matrix of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    uv_homogeneous = np.stack((u, v, np.ones_like(u)), axis=-1).reshape(-1, 3)

    # Invert the intrinsic matrix
    inv_intrinsics = np.linalg.inv(intrinsics)

    # Convert depth to the camera coordinate system
    points_cam_homogeneous = np.dot(uv_homogeneous, inv_intrinsics.T) * depth.flatten()[:, np.newaxis]

    # Convert to 3D homogeneous coordinates
    points_cam_homogeneous = np.concatenate((points_cam_homogeneous, np.ones((len(points_cam_homogeneous), 1))), axis=1)

    # Apply the rotation and translation to get the 3D point cloud in the world coordinate system
    extrinsics = np.hstack((rotation, translation[:, np.newaxis]))
    pointcloud = np.dot(points_cam_homogeneous, extrinsics.T)

    # Reshape the point cloud back to the original depth map shape
    pointcloud = pointcloud[:, :3].reshape(height, width, 3)

    return pointcloud


def point_cloud_to_depth_map(point_cloud, img_shape):
    """
    Project a point cloud into a depth map.

    point_cloud: numpy array of shape (N, 3) with 3D coordinates in the camera frame
    K: intrinsic camera matrix
    img_shape: tuple with the shape of the depth map (height, width)
    """
    K = get_intrinsics(img_shape[0], img_shape[1])
    # Project 3D points to 2D image coordinates
    points_2d = K @ point_cloud.T
    points_2d /= points_2d[2, :]
    points_2d = points_2d[:2, :].T

    # Round the 2D points to integers
    points_2d = np.round(points_2d).astype(int)

    # Filter out points outside the image dimensions
    valid_points = (0 <= points_2d[:, 0]) & (points_2d[:, 0] < img_shape[1]) & \
          (0 <= points_2d[:, 1]) & (points_2d[:, 1] < img_shape[0])
    points_2d = points_2d[valid_points]
    point_cloud = point_cloud[valid_points]

    # Create a depth map and fill in the depths at the corresponding 2D points
    depth_map = np.zeros(img_shape, dtype=np.float32)
    depth_map[points_2d[:, 1], points_2d[:, 0]] = point_cloud[:, 2]

    return depth_map


def main(args):

    model = mcc_model.get_mcc_model(
        occupancy_weight=1.0,
        rgb_weight=0.01,
        args=args,
    ).cuda()

    misc.load_model(args=args, model_without_ddp=model, optimizer=None, loss_scaler=None)

    bgr = cv2.imread(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    rr.set_time_sequence("num_passes", 0)
    rr.log("input-image", rr.Image(rgb))
    
    # seen in this context means the points that are visible in the image
    seen_rgb = (torch.tensor(bgr).float() / 255)[..., [2, 1, 0]]
    H, W = seen_rgb.shape[:2]
    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[H, W],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)


    if args.seg is None:
        # Generate mask using Segment Anything Mask (SAM)
        sam = create_sam("vit_b", "cuda")
        predictor = SamPredictor(sam)
        predictor.set_image(rgb)
        
        bbox = np.array(args.bbox_for_seg)

        rr.log(
            "input-image/bbox-for-seg",
            rr.Boxes2D(
                mins=bbox[:2],
                sizes=bbox[2:] - bbox[:2],
                colors=(255, 0, 0),
            )
        )

        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=bbox[None, :],
            multimask_output=False,
        )
        seg = masks[0].astype(np.uint8)
    else:
        seg = cv2.imread(args.seg, cv2.IMREAD_UNCHANGED)

    mask = torch.tensor(cv2.resize(seg, (W, H))).bool()
    rr.log("mask", rr.Tensor(mask.float()))

    if args.point_cloud is None:
        # Get depth map and convert to point cloud
        depth_model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_N", pretrained=True).to("cuda").eval()
        depth = depth_model.infer(seen_rgb.permute(2, 0, 1)[None].cuda())
        depth = depth[0].permute(1, 2, 0)
        depth = depth.cpu().detach().numpy().squeeze()
        rr.log("depth", rr.DepthImage(depth))
        seen_xyz = backproject_depth_to_pointcloud(depth)
        seen_xyz = torch.tensor(seen_xyz).float()
    else:
        obj = load_obj(args.point_cloud)
        # Verts from OBJ file reshaped to image size
        seen_xyz = obj[0].reshape(H, W, 3)
        depth = point_cloud_to_depth_map(obj[0].numpy(), (H, W))
        rr.log("depth", rr.DepthImage(depth))
    seen_xyz[~mask] = float('inf')

    rr.log(
        "input_points",
        rr.Points3D(positions=seen_xyz[mask], colors=seen_rgb[mask].numpy())
    )

    seen_xyz = normalize(seen_xyz)

    bottom, right = mask.nonzero().max(dim=0)[0]
    top, left = mask.nonzero().min(dim=0)[0]

    bottom = bottom + 40
    right = right + 40
    top = max(top - 40, 0)
    left = max(left - 40, 0)

    seen_xyz = seen_xyz[top:bottom+1, left:right+1]
    seen_rgb = seen_rgb[top:bottom+1, left:right+1]

    seen_xyz = pad_image(seen_xyz, float('inf'))
    seen_rgb = pad_image(seen_rgb, 0)

    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[800, 800],
        mode="bilinear",
        align_corners=False,
    )

    seen_xyz = torch.nn.functional.interpolate(
        seen_xyz.permute(2, 0, 1)[None],
        size=[112, 112],
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1)

    samples = [
        [seen_xyz, seen_rgb],
        [torch.zeros((20000, 3)), torch.zeros((20000, 3))],
    ]
    run_viz(model, samples, "cuda", args)


if __name__ == '__main__':
    parser = main_mcc.get_args_parser()
    parser.add_argument('--image', default='demo/spyro.jpg', type=str, help='input image file')
    parser.add_argument('--point_cloud', type=str, help='input obj file')
    parser.add_argument('--seg', type=str, help='input obj file')
    parser.add_argument(
        '--bbox-for-seg',
        default=[27,44,412,595],
        type=int,
        nargs='+',
        help='coordinates for bounding box to segment object, has format of xyxy')
    parser.add_argument('--granularity', default=0.05, type=float, help='output granularity')
    parser.add_argument('--score_thresholds', default=[0.1, 0.2, 0.3, 0.4, 0.5], type=float, nargs='+', help='score thresholds')
    parser.add_argument('--temperature', default=0.1, type=float, help='temperature for color prediction.')

    parser.set_defaults(eval=True)
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "MCC")

    # check that the checkpoint exists
    checkpoint_path = Path("checkpoint") / "co3dv2_all_categories.pth"
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir()
        download_with_progress(MODEL_URLS["mcc"], checkpoint_path)

    args.resume = str(checkpoint_path)
    args.viz_granularity = args.granularity

    main(args)
    rr.script_teardown(args)

