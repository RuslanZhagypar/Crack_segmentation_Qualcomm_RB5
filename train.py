import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import ColorJitter
from torch.optim.lr_scheduler import OneCycleLR

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

from fast_scnn import FastSCNN


# --------------------------
# Utility: Center‐crop + resize
# --------------------------
def crop_center_square_and_resize(
    img: Image.Image,
    size: int = 576,
    resample=Image.LANCZOS
) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    sq = img.crop((left, top, left + side, top + side))
    return sq.resize((size, size), resample)



import torchvision.transforms.functional as TF
import random
from PIL import ImageFilter

import os, random, numpy as np
from dataclasses import dataclass
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

# --------------------------
# Utility: center-square + resize (you already have this)
# --------------------------
# def crop_center_square_and_resize(...):  # keep your version

@dataclass
# class AugCfg:
#     p_hflip: float = 0.5
#     p_vflip: float = 0.1
#     p_affine: float = 0.4
#     rot_deg: float = 10.0          # milder than 20
#     translate: float = 0.05        # fraction of width/height
#     scale_min: float = 0.9
#     scale_max: float = 1.1
#     shear_deg: float = 5.0
#     p_persp: float = 0.2
#     persp_distort: float = 0.15    # milder than 0.4
#     p_jitter: float = 0.5
#     jitter_bcsj: tuple = (0.2, 0.2, 0.2, 0.05)  # brightness/contrast/saturation/hue
#     p_gray: float = 0.05
#     p_blur: float = 0.15
#     blur_min: float = 0.3
#     blur_max: float = 1.2

# class AugCfg:
#     p_hflip:  float = 0.6
#     p_vflip:  float = 0.12
#     p_affine: float = 0.5
#     rot_deg:  float = 18.0
#     translate: float = 0.08
#     scale_min: float = 0.85
#     scale_max: float = 1.18
#     shear_deg: float = 8.0
#     p_persp: float = 0.3
#     persp_distort: float = 0.2
#     p_jitter: float = 0.55
#     jitter_bcsj: tuple = (0.35, 0.35, 0.35, 0.08)
#     p_gray: float = 0.07
#     p_blur: float = 0.18
#     blur_min: float = 0.35
#     blur_max: float = 0.9

# class CenterDataset(Dataset):
#     def __init__(self, root_dir="dataset_v3/train", size=576, aug=True, aug_cfg: AugCfg | None = None):
#         self.root_dir = root_dir
#         self.size = size
#         self.aug = aug
#         self.cfg = aug_cfg or AugCfg()

#         self.files = sorted([
#             f for f in os.listdir(root_dir)
#             if f.endswith(".jpg") and os.path.exists(
#                 os.path.join(root_dir, f.replace(".jpg", "_mask.png"))
#             )
#         ])

#         # image-only photometric transforms (applied after geometry)
#         c = self.cfg
#         self.photo_tf = transforms.Compose([
#             transforms.ColorJitter(*c.jitter_bcsj) if c.p_jitter > 0 else transforms.Lambda(lambda x: x),
#         ])

#         self.to_tensor_norm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406],
#                                  [0.229, 0.224, 0.225])
#         ])

#     def __len__(self):
#         return len(self.files)

#     def _geom_params(self, w, h):
#         c = self.cfg
#         params = {}

#         # flips
#         params["do_hflip"] = (random.random() < c.p_hflip)
#         params["do_vflip"] = (random.random() < c.p_vflip)

#         # affine
#         params["do_affine"] = (random.random() < c.p_affine)
#         if params["do_affine"]:
#             params["angle"] = random.uniform(-c.rot_deg, c.rot_deg)
#             params["translate"] = (int(random.uniform(-c.translate, c.translate) * w),
#                                    int(random.uniform(-c.translate, c.translate) * h))
#             params["scale"] = random.uniform(c.scale_min, c.scale_max)
#             params["shear"] = random.uniform(-c.shear_deg, c.shear_deg)

#         # perspective
#         params["do_persp"] = (random.random() < c.p_persp)
#         if params["do_persp"]:
#             d = c.persp_distort
#             sp = [(0,0), (w,0), (w,h), (0,h)]
#             ep = [
#                 (random.uniform(0, d)*w,                 random.uniform(0, d)*h),
#                 (w - random.uniform(0, d)*w,            random.uniform(0, d)*h),
#                 (w - random.uniform(0, d)*w,            h - random.uniform(0, d)*h),
#                 (random.uniform(0, d)*w,                h - random.uniform(0, d)*h),
#             ]
#             params["startpoints"] = sp
#             params["endpoints"] = ep
#         return params

#     def _apply_geom(self, img: Image.Image, msk: Image.Image, params):
#         # flips
#         if params["do_hflip"]:
#             img = TF.hflip(img); msk = TF.hflip(msk)
#         if params["do_vflip"]:
#             img = TF.vflip(img); msk = TF.vflip(msk)

#         # affine (use different interpolation/fill)
#         if params["do_affine"]:
#             img = TF.affine(
#                 img,
#                 angle=params["angle"],
#                 translate=params["translate"],
#                 scale=params["scale"],
#                 shear=params["shear"],
#                 interpolation=transforms.InterpolationMode.BILINEAR,
#                 fill=0,
#             )
#             msk = TF.affine(
#                 msk,
#                 angle=params["angle"],
#                 translate=params["translate"],
#                 scale=params["scale"],
#                 shear=params["shear"],
#                 interpolation=transforms.InterpolationMode.NEAREST,
#                 fill=0,
#             )

#         # perspective
#         if params["do_persp"]:
#             img = TF.perspective(
#                 img, params["startpoints"], params["endpoints"],
#                 interpolation=transforms.InterpolationMode.BILINEAR, fill=0
#             )
#             msk = TF.perspective(
#                 msk, params["startpoints"], params["endpoints"],
#                 interpolation=transforms.InterpolationMode.NEAREST, fill=0
#             )
#         return img, msk

#     def __getitem__(self, idx):
#         fname = self.files[idx]
#         stem = fname[:-4]
#         img = Image.open(os.path.join(self.root_dir, fname)).convert("RGB")
#         msk = Image.open(os.path.join(self.root_dir, stem + "_mask.png")).convert("L")

#         # center-square + resize
#         img = crop_center_square_and_resize(img, self.size, Image.BILINEAR)
#         msk = crop_center_square_and_resize(msk, self.size, Image.NEAREST)

#         # consistent geometry for img & mask
#         if self.aug:
#             params = self._geom_params(img.width, img.height)
#             img, msk = self._apply_geom(img, msk, params)

#             # mild photometric on image only
#             if random.random() < self.cfg.p_jitter:
#                 img = self.photo_tf(img)

#             if random.random() < self.cfg.p_gray:
#                 img = TF.rgb_to_grayscale(img, num_output_channels=3)

#             if random.random() < self.cfg.p_blur:
#                 img = img.filter(ImageFilter.GaussianBlur(
#                     radius=random.uniform(self.cfg.blur_min, self.cfg.blur_max))
#                 )

#         # to tensor
#         img_t = self.to_tensor_norm(img)
#         msk_np = np.array(msk, dtype=np.uint8)
#         # binarize mask robustly
#         if msk_np.max() > 1:
#             msk_np = (msk_np > 127).astype(np.uint8)
#         msk_t = torch.from_numpy(msk_np).long()
#         return img_t, msk_t



# # --------------------------
# # Datasets
# # --------------------------
class CenterDataset(Dataset):
    """
    Expects images as *.jpg and masks as *_mask.png in the same folder.
    """
    def __init__(self, root_dir="dataset_v3/train", size=576, aug=True):
        self.root_dir = root_dir
        self.size = size
        self.aug = aug

        self.files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".jpg") and os.path.exists(
                os.path.join(root_dir, f.replace(".jpg", "_mask.png"))
            )
        ])

        # photometric (image only)
        self.photo_tf = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        stem = fname[:-4]

        img = Image.open(os.path.join(self.root_dir, fname)).convert("RGB")
        msk = Image.open(os.path.join(self.root_dir, stem + "_mask.png")).convert("L")

        # center-square + resize (keeps both aligned)
        img = crop_center_square_and_resize(img, self.size, Image.BILINEAR)
        msk = crop_center_square_and_resize(msk, self.size, Image.NEAREST)

        if self.aug:
            # ----- geometry: apply to BOTH -----
            if random.random() < 0.5:
                img = TF.hflip(img); msk = TF.hflip(msk)
            if random.random() < 0.1:
                img = TF.vflip(img); msk = TF.vflip(msk)
            if random.random() < 0.5:
                angle = random.uniform(-12, 12)
                img = TF.rotate(img, angle,
                                interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
                msk = TF.rotate(msk, angle,
                                interpolation=transforms.InterpolationMode.NEAREST,  fill=0)

            # ----- photometric: image only -----
            if random.random() < 0.5:
                img = self.photo_tf(img)
            if random.random() < 0.1:
                img = TF.rgb_to_grayscale(img, num_output_channels=3)
            if random.random() < 0.15:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

        img_t = self.to_tensor_norm(img)

        msk_np = np.array(msk, dtype=np.uint8)
        if msk_np.max() > 1:  # handle {0,255}
            msk_np = (msk_np > 127).astype(np.uint8)
        msk_t = torch.from_numpy(msk_np).long()

        return img_t, msk_t

# class CenterDataset(Dataset):
#     """
#     Expects images as *.jpg and masks as *_mask.png in the same folder.
#     """
#     def __init__(self, root_dir="dataset_v3/train", size=576, aug=True):
#         self.root_dir = root_dir
#         self.size = size
#         self.aug = aug

#         self.files = sorted([
#             f for f in os.listdir(root_dir)
#             if f.endswith(".jpg") and os.path.exists(
#                 os.path.join(root_dir, f.replace(".jpg", "_mask.png"))
#             )
#         ])

#         aug_tf = transforms.Compose([
#             transforms.RandomApply([transforms.RandomRotation(15, fill=0)], p=0.5),
#             transforms.RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#             transforms.RandomGrayscale(p=0.1),
#         ])

#         self.img_tf = transforms.Compose([
#             aug_tf if aug else transforms.Lambda(lambda x: x),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 [0.485, 0.456, 0.406],
#                 [0.229, 0.224, 0.225]
#             )
#         ])

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         fname = self.files[idx]
#         stem = fname[:-4]  # remove .jpg

#         img = Image.open(os.path.join(self.root_dir, fname)).convert("RGB")
#         msk = Image.open(os.path.join(self.root_dir, stem + "_mask.png")).convert("L")

#         img_c = crop_center_square_and_resize(img, self.size, Image.BILINEAR)
#         msk_c = crop_center_square_and_resize(msk, self.size, Image.NEAREST)

#         img_t = self.img_tf(img_c)
#         msk_t = torch.from_numpy(np.array(msk_c, dtype=np.uint8)).long()
#         # ensure labels are {0,1}
#         if msk_t.max() > 1:
#             msk_t = (msk_t > 127).long()

#         return img_t, msk_t


# --------------------------
# Losses
# --------------------------
# class HybridLoss(nn.Module):
#     def __init__(self, device, ce_weight=(1.0, 50.0), smooth=0.5):
#         super().__init__()
#         weight = torch.tensor(ce_weight, dtype=torch.float32, device=device)
#         self.ce = nn.CrossEntropyLoss(weight=weight)
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         ce_loss = self.ce(logits, targets)
#         probs = torch.softmax(logits, dim=1)
#         C = logits.size(1)
#         one_hot = F.one_hot(targets, C).permute(0, 3, 1, 2).float()
#         dims = (0, 2, 3)
#         inter = torch.sum(probs * one_hot, dims)
#         union = torch.sum(probs + one_hot, dims)
#         dice_score = (2 * inter + self.smooth) / (union + self.smooth)
#         dice_loss = 1 - dice_score.mean()
#         return ce_loss + dice_loss

class HybridLoss(nn.Module):
    def __init__(self, ce_weight=(1.,1.), smooth=1.0, alpha=0.3, beta=0.7, gamma=1.33):
        super().__init__()
        self.register_buffer("w", torch.tensor(ce_weight, dtype=torch.float32))
        self.smooth, self.alpha, self.beta, self.gamma = smooth, alpha, beta, gamma
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.w.to(logits.device, logits.dtype))
        p1 = logits.softmax(1)[:,1:2]; g1 = (targets==1).float().unsqueeze(1)
        tp = (p1*g1).sum((0,2,3)); fp = (p1*(1-g1)).sum((0,2,3)); fn = ((1-p1)*g1).sum((0,2,3))
        t = (tp+self.smooth)/(tp+self.alpha*fp+self.beta*fn+self.smooth)
        dice_like = t.clamp_min(1e-6).pow(1.0/self.gamma)  # focal Tversky
        return ce + (1.0 - dice_like.mean())


#  #----------------------------------------------------------

 #----------------------------------------------------------       

# --------------------------
# Inference helper
# --------------------------
@torch.no_grad()
def infer_single_image(model, image_path, size=576, device="cpu"):
    img = Image.open(image_path).convert("RGB")
    inp = crop_center_square_and_resize(img, size=size, resample=Image.BILINEAR)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])
    x = tf(inp).unsqueeze(0).to(device)
    out = model(x)
    out = out[0] if isinstance(out, (tuple, list)) else out
    mask = torch.argmax(out, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    return inp, mask


def compute_binary_dice(pred_np: np.ndarray, gt_np: np.ndarray, eps=1e-6):
    pred = (pred_np == 1).astype(np.uint8)
    gt = (gt_np == 1).astype(np.uint8)
    inter = np.logical_and(pred, gt).sum()
    uni = np.logical_or(pred, gt).sum()
    dice = (2 * inter) / (inter + uni + eps)
    return float(dice)


def overlay_mask_on_image(pil_img: Image.Image, mask_np: np.ndarray, alpha=0.5):
    """Return an overlay PIL image."""
    # ensure HxW
    mask_np = (mask_np.astype(np.uint8) * 255)  # assume classes {0,1}
    mask_rgb = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)
    # color for class 1 (crack): red
    mask_rgb[mask_np > 127] = np.array([255, 0, 0], dtype=np.uint8)
    base = np.array(pil_img).astype(np.float32)
    over = mask_rgb.astype(np.float32)
    blended = (1 - alpha) * base + alpha * over
    blended = blended.clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended)


# --------------------------
# Main
# --------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # p.add_argument("--train_root", type=str, default="DATASETS/New_dataset_12/train/all_files")
    p.add_argument("--train_root", type=str, default="DATASETS/dataset_v3/train")
    # p.add_argument("--train_root", type=str, default="roboflow_train_crack/images")
    p.add_argument("--valid_root", type=str, default="DATASETS/dataset_v3/valid")
    # p.add_argument("--val_img_name", type=str, default="sample_11_from_video.jpg")
    # p.add_argument("--val_mask_name", type=str, default="sample_11_from_video_mask.png")  # as you asked
    p.add_argument("--val_img_name", type=str, default="frame_20250829_143907.jpg")
    p.add_argument("--val_mask_name", type=str, default="frame_20250829_143907_mask.png")
    # p.add_argument("--val_img_name", type=str, default="val_1.jpg")
    # p.add_argument("--val_mask_name", type=str, default="val_1.png")  # as you asked
    # p.add_argument("--val_img_name", type=str, default="IMG_4444.jpg")
    # p.add_argument("--val_mask_name", type=str, default="IMG_4444_mask.png")  # as you asked
    p.add_argument("--epochs", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--size", type=int, default=576)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--pretrained", type=str, default="fast_scnn_bn_pretrained_BN_val1000.pth")
    p.add_argument("--save_dir", type=str, default="checkpoints_fastscnn_accel555")
    p.add_argument("--images_dir", type=str, default="Images_accel555")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.images_dir, exist_ok=True)

    # Datasets / Loaders
    train_ds = CenterDataset(root_dir=args.train_root, size=args.size, aug=True)
    valid_ds = CenterDataset(root_dir=args.valid_root, size=args.size, aug=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        drop_last=True
    )

    # Model/opt/sched
    model = FastSCNN(num_classes=2, aux=False)
    # # load pretrained safely
    # if args.pretrained and os.path.isfile(args.pretrained):
    #     state = torch.load(args.pretrained, map_location="cpu")
    #     missing, unexpected = model.load_state_dict(state, strict=False)
    #     # (Optional) you can accelerator.print them
    #     accelerator.print(f"Loaded pretrained. Missing: {missing}, Unexpected: {unexpected}")
    # else:
    #     accelerator.print(f"Pretrained weights not found at {args.pretrained}. Training from scratch.")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # steps_per_epoch = len(train_loader)  # number of batches in training set
    # total_steps = args.epochs * steps_per_epoch
    # # OneCycleLR scheduler
    # scheduler = OneCycleLR(
    #     optimizer,
    #     max_lr=args.lr,                  # peak LR
    #     total_steps=total_steps,      # total number of steps
    #     pct_start=0.3,                # % of cycle spent increasing LR
    #     anneal_strategy='cos',        # cosine annealing
    #     div_factor=25.0,              # initial LR = max_lr / div_factor
    #     final_div_factor=1e4,         # min LR = initial LR / final_div_factor
    #     three_phase=False
    # )
    iters = len(train_loader) * args.epochs
    def poly(it): return (1 - it / max(1, iters)) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=poly)

    # criterion = HybridLoss(device=accelerator.device)
    def compute_ce_weights(ds, eps=1e-6, fg_boost=1.5):
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        pos=0; tot=0
        for _, m in loader:
            mm = m.numpy()
            pos += (mm==1).sum(); tot += mm.size
        neg = max(tot-pos, 1)
        w_bg = tot/(2*neg+eps); w_fg = tot/(2*max(pos,1)+eps)*fg_boost
        s = (w_bg+w_fg)/2; w_bg/=s; w_fg/=s
        return float(np.clip(w_bg,0.05,50.)), float(np.clip(w_fg,0.05,50.))
    # usage
    w_bg, w_fg = compute_ce_weights(CenterDataset(args.train_root, args.size, aug=False))
    criterion = HybridLoss(ce_weight=(w_bg, w_fg), alpha=0.3, beta=0.7, gamma=1.33)




    # Prepare with accelerate
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    best_dice = 0.0
    train_loss_history = []
    train_epoch_history = []   # << NEW
    val_dice_history = []
    val_epoch_history  = []    # << NEW

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, disable=not accelerator.is_local_main_process, leave=False)
        for imgs, masks in pbar:
            imgs = imgs.to(accelerator.device, non_blocking=True)
            masks = masks.to(accelerator.device, non_blocking=True)

            logits = model(imgs)
            logits = logits[0] if isinstance(logits, (tuple, list)) else logits
            loss = criterion(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch} | loss {loss.item():.4f}")

        # scheduler.step()
        avg_loss = total_loss / max(1, len(train_loader))
        accelerator.print(f"[Epoch {epoch:04d}] Train Loss: {avg_loss:.5f}")
        train_loss_history.append(avg_loss)      # track loss
        train_epoch_history.append(epoch)        # << NEW

        # --------------------------
        # Validation + Visualization
        # --------------------------
        if (epoch % args.val_every == 0) or (epoch == args.epochs):
            model.eval()

            # 1) quantitative Dice on the whole valid set (gather not needed if small; simple local loop)
            dices = []
            with torch.no_grad():
                vbar = tqdm(valid_ds, disable=not accelerator.is_local_main_process, leave=False)
                for i in range(len(valid_ds)):
                    img_t, gt_t = valid_ds[i]
                    x = img_t.unsqueeze(0).to(accelerator.device)
                    out = model(x)
                    out = out[0] if isinstance(out, (tuple, list)) else out
                    pred = torch.argmax(out, dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
                    gt = gt_t.numpy().astype(np.uint8)
                    dice = compute_binary_dice(pred, gt)
                    dices.append(dice)

            # reduce across processes if needed
            if accelerator.num_processes > 1:
                # pad to same length then gather+trim
                length = torch.tensor([len(dices)], device=accelerator.device)
                lengths = accelerator.gather(length).cpu().tolist()
                max_len = max(lengths)
                pad_val = -1.0
                local = torch.full((max_len,), pad_val, device=accelerator.device, dtype=torch.float32)
                if len(dices) > 0:
                    local[:len(dices)] = torch.tensor(dices, device=accelerator.device)
                gathered = accelerator.gather(local)
                gathered = gathered.cpu().numpy()
                # filter out padding
                dices_all = [float(x) for x in gathered if x >= 0]
            else:
                dices_all = dices

            mean_dice = float(np.mean(dices_all)) if len(dices_all) else 0.0
            accelerator.print(f"[Epoch {epoch:04d}] Valid mean Dice: {mean_dice:.4f}")
            val_dice_history.append(mean_dice)       # track dice
            val_epoch_history.append(epoch)          # << NEW

            # 2) visualization on explicit val_1.jpg / val_1.png (from dataset_v3/valid)
            if accelerator.is_local_main_process:
                val_img_path = os.path.join(args.valid_root, args.val_img_name)
                val_mask_path = os.path.join(args.valid_root, args.val_mask_name)

                # prediction
                inp_pil, pred_np = infer_single_image(
                    model, val_img_path, size=args.size, device=accelerator.device
                )
                # ground-truth (explicit file you requested)
                gt_pil = Image.open(val_mask_path).convert("L")
                gt_pil = crop_center_square_and_resize(gt_pil, size=args.size, resample=Image.NEAREST)
                gt_np = np.array(gt_pil, dtype=np.uint8)
                if gt_np.max() > 1:
                    gt_np = (gt_np > 127).astype(np.uint8)

                # overlays
                pred_overlay = overlay_mask_on_image(inp_pil, pred_np, alpha=0.45)
                gt_overlay = overlay_mask_on_image(inp_pil, gt_np, alpha=0.45)

                # compute dice on this specific pair too
                val1_dice = compute_binary_dice(pred_np, gt_np)

                # Save a 4-panel figure
                fig, axes = plt.subplots(1, 4, figsize=(18, 4))
                axes[0].imshow(inp_pil);         axes[0].set_title("Input");                axes[0].axis("off")
                axes[1].imshow(gt_np, cmap="gray"); axes[1].set_title("GT mask");           axes[1].axis("off")
                axes[2].imshow(pred_np, cmap="gray"); axes[2].set_title("Pred mask");       axes[2].axis("off")
                axes[3].imshow(pred_overlay);     axes[3].set_title(f"Overlay (Dice={val1_dice:.3f})"); axes[3].axis("off")
                plt.tight_layout()
                out_path = os.path.join(args.images_dir, f"val_epoch_{epoch:04d}.png")
                plt.savefig(out_path, dpi=150)
                plt.close()
                accelerator.print(f"Saved visualization to {out_path}")

                # ---------- NEW: metrics plot with epoch-aligned x-axis ----------
                fig2, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))

                # Train loss vs epoch (left y-axis)
                ax1.plot(train_epoch_history, train_loss_history, label="Train Loss")
                ax1.set_xlabel("Epoch")
                ax1.set_ylabel("Train Loss")
                ax1.grid(True, alpha=0.3)

                # Val dice vs epoch (right y-axis), plotted only at validation epochs
                ax2 = ax1.twinx()
                ax2.plot(val_epoch_history, val_dice_history, marker="o", linestyle="-", label="Val Dice")
                ax2.set_ylabel("Val Dice")
                ax2.set_ylim(0.0, 1.0)  # dice in [0,1]

                # Unified legend
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

                ax1.set_title("Training Loss & Validation Dice over Epochs")

                plot_path = os.path.join(args.images_dir, "training_plot.png")
                plt.tight_layout()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                accelerator.print(f"Updated training plot at {plot_path}")
                # ----------------------------------------------------------------- 

            # Save best (main process only)
            if accelerator.is_local_main_process and mean_dice > best_dice:
                best_dice = mean_dice
                save_path = os.path.join(args.save_dir, f"best_fastscnn_epoch{epoch:04d}.pth")
                accelerator.print(f"✅ New best Dice {best_dice:.4f}. Saving to {save_path}")
                accelerator.save(model.state_dict(), save_path)

            model.train()

    accelerator.print("Training complete.")


if __name__ == "__main__":
    main()
