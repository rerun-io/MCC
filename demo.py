# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2
from tqdm import tqdm

import torch
from pytorch3d.io.obj_io import load_obj

import main_mcc
import mcc_model
import util.misc as misc
import rerun as rr
import requests
from pathlib import Path
from engine_mcc import prepare_data

MODEL_URL: str = "https://dl.fbaipublicfiles.com/MCC/co3dv2_all_categories.pth"

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

def occupancy_to_points(occupancy, xyz, color, threshold=0.1):

    occupancy = torch.nn.Sigmoid()(occupancy)
    pos = occupancy > threshold
    points = xyz[pos].reshape((-1, 3))
    features = color[None][pos].reshape((-1, 3))
    good_points = points[:, 0] != -100

    if good_points.sum() == 0:
        return
    
    rr.log_points(
        "world/points",
        positions = points[good_points].cpu(),
        colors = features[good_points].cpu())

def run_viz(model, samples, device, args, prefix):
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
            occupancy_to_points(
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


def main(args):

    model = mcc_model.get_mcc_model(
        occupancy_weight=1.0,
        rgb_weight=0.01,
        args=args,
    ).cuda()

    misc.load_model(args=args, model_without_ddp=model, optimizer=None, loss_scaler=None)

    rgb = cv2.imread(args.image)
    obj = load_obj(args.point_cloud)

    rr.set_time_sequence("num_passes", 0)
    rr.log_image("input", rgb[..., ::-1])

    seen_rgb = (torch.tensor(rgb).float() / 255)[..., [2, 1, 0]]
    H, W = seen_rgb.shape[:2]
    seen_rgb = torch.nn.functional.interpolate(
        seen_rgb.permute(2, 0, 1)[None],
        size=[H, W],
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0)

    seen_xyz = obj[0].reshape(H, W, 3)
    seg = cv2.imread(args.seg, cv2.IMREAD_UNCHANGED)
    mask = torch.tensor(cv2.resize(seg, (W, H))).bool()
    rr.log_tensor("mask", mask.float())
    seen_xyz[~mask] = float('inf')
    rr.log_tensor("seen_xyz", seen_xyz)

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
    run_viz(model, samples, "cuda", args, prefix=args.output)


if __name__ == '__main__':
    parser = main_mcc.get_args_parser()
    parser.add_argument('--image', default='demo/quest2.jpg', type=str, help='input image file')
    parser.add_argument('--point_cloud', default='demo/quest2.obj', type=str, help='input obj file')
    parser.add_argument('--seg', default='demo/quest2_seg.png', type=str, help='input obj file')
    parser.add_argument('--output', default='demo/output', type=str, help='output path')
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
        download_with_progress(MODEL_URL, checkpoint_path)

    args.resume = str(checkpoint_path)
    args.viz_granularity = args.granularity

    main(args)
    rr.script_teardown(args)

