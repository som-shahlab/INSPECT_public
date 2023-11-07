import torch
import torch.nn as nn
import torchvision
import timm

from torchvision import models as model_2d


class Identity(nn.Module):
    """Identity layer to replace last fully connected layer"""

    def forward(self, x):
        return x


def resnext_101_sup_ct(**kwargs):
    model = model_2d.resnext101_32x8d(pretrained=True)
    features_dims = model.fc.in_features
    model.fc = Identity()

    checkpoint = torch.load("/home/mschuang/radfusion3.0/ckpt/resnext101_ct.ckpt")
    ckpt = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    msg = model.load_state_dict(ckpt, strict=False)
    print("=" * 80)
    print(msg)
    print("=" * 80)
    return model, features_dims


def resnetv2_101_sup(**kwargs):
    model = timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)
    model.head.fc = Identity()
    return model, 6144


def resnetv2_101_ct(**kwargs):
    model = timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)
    model.head.fc = Identity()

    # checkpoint = torch.load('/home/mschuang/radfusion3.0/ckpt/resnetv2_ct.ckpt')
    checkpoint = torch.load(
        "/share/pi/nigam/projects/zphuo/data/PE/inspect/image_modality/ckpt/resnetv2_ct.ckpt"
    )
    ckpt = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    msg = model.load_state_dict(ckpt, strict=False)
    print("=" * 80)
    print(msg)
    print("=" * 80)
    return model, 6144


################################################################################
# Vision Transformers
################################################################################


def vit_base_16_dinov1_ct(**kwargs):
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    checkpoint = torch.load("/home/mschuang/radfusion3.0/ckpt/dinov1_ct.ckpt")
    ckpt = {
        k.replace("student.backbone.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if "student.backbone" in k
    }
    model.load_state_dict(ckpt)
    model.head = Identity()
    return model, model.embed_dim


def vit_base_16_dinov1(**kwargs):
    model = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
    model.head = Identity()
    return model, model.embed_dim


def vit_base_14_dinov2(**kwargs):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.head = Identity()
    return model, model.embed_dim


def vit_base_16_sup(**kwargs):
    model = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True
    )
    model.head = Identity()
    return model, model.embed_dim


def vit_base_16_clip(**kwargs):
    model = timm.create_model(
        "vit_base_patch16_clip_224.openai_ft_in12k_in1k", pretrained=True
    )
    model.head = Identity()
    return model, model.embed_dim


def vit_base_16_swin(**kwargs):
    model = timm.create_model("swinv2_base_window16_256.ms_in1k", pretrained=True)
    model.head = Identity()
    return model, model.embed_dim


def convnext_base_clip(**kwargs):
    model = timm.create_model("convnext_base.clip_laion2b", pretrained=True)
    model.head.fc = Identity()
    return model, 1024


def swinv2_base_sup(**kwargs):
    model = timm.create_model(
        "swinv2_base_window12to16_192to256_22kft1k", pretrained=True
    )
    model.head.fc = Identity()
    return model, 1024


def swinv2_base_ct(**kwargs):
    model = timm.create_model(
        "swinv2_base_window12to16_192to256_22kft1k", pretrained=True
    )
    model.head.fc = Identity()

    checkpoint = torch.load("/home/mschuang/radfusion3.0/ckpt/swint_ct.ckpt")
    ckpt = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    msg = model.load_state_dict(ckpt, strict=False)
    print("=" * 80)
    print(msg)
    print("=" * 80)

    return model, 1024
