import numpy as np
from .base_transforms import jitter, scaling, rotation, permutation
from .warping_transforms import magnitude_warp, time_warp, window_slice, window_warp
from .dtw_transforms import (
    spawner, wdba, random_guided_warp, random_guided_warp_shape,
    discriminative_guided_warp, discriminative_guided_warp_shape
)


def run_augmentation(x, y, args):
    print("Augmenting %s"%args.data)
    np.random.seed(args.seed)
    x_aug = x
    y_aug = y
    if args.augmentation_ratio > 0:
        augmentation_tags = "%d"%args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            x_temp, augmentation_tags = augment(x, y, args)
            x_aug = np.append(x_aug, x_temp, axis=0)
            y_aug = np.append(y_aug, y, axis=0)
            print("Round %d: %s done"%(n, augmentation_tags))
        if args.extra_tag:
            augmentation_tags += "_"+args.extra_tag
    else:
        augmentation_tags = args.extra_tag
    return x_aug, y_aug, augmentation_tags


def run_augmentation_single(x, y, args):
    np.random.seed(args.seed)

    x_aug = x
    y_aug = y

    if len(x.shape)<3:
        x_input = x[np.newaxis,:]
    elif len(x.shape)==3:
        x_input = x
    else:
        raise ValueError("Input must be (batch_size, sequence_length, num_channels) dimensional")

    if args.augmentation_ratio > 0:
        augmentation_tags = "%d"%args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            x_aug, augmentation_tags = augment(x_input, y, args)
        if args.extra_tag:
            augmentation_tags += "_"+args.extra_tag
    else:
        augmentation_tags = args.extra_tag

    if(len(x.shape)<3):
        x_aug = x_aug.squeeze(0)
    return x_aug, y_aug, augmentation_tags


def augment(x, y, args):
    augmentation_tags = ""
    if args.jitter:
        x = jitter(x)
        augmentation_tags += "_jitter"
    if args.scaling:
        x = scaling(x)
        augmentation_tags += "_scaling"
    if args.rotation:
        x = rotation(x)
        augmentation_tags += "_rotation"
    if args.permutation:
        x = permutation(x)
        augmentation_tags += "_permutation"
    if args.randompermutation:
        x = permutation(x, seg_mode="random")
        augmentation_tags += "_randomperm"
    if args.magwarp:
        x = magnitude_warp(x)
        augmentation_tags += "_magwarp"
    if args.timewarp:
        x = time_warp(x)
        augmentation_tags += "_timewarp"
    if args.windowslice:
        x = window_slice(x)
        augmentation_tags += "_windowslice"
    if args.windowwarp:
        x = window_warp(x)
        augmentation_tags += "_windowwarp"
    if args.spawner:
        x = spawner(x, y)
        augmentation_tags += "_spawner"
    if args.dtwwarp:
        x = random_guided_warp(x, y)
        augmentation_tags += "_rgw"
    if args.shapedtwwarp:
        x = random_guided_warp_shape(x, y)
        augmentation_tags += "_rgws"
    if args.wdba:
        x = wdba(x, y)
        augmentation_tags += "_wdba"
    if args.discdtw:
        x = discriminative_guided_warp(x, y)
        augmentation_tags += "_dgw"
    if args.discsdtw:
        x = discriminative_guided_warp_shape(x, y)
        augmentation_tags += "_dgws"
    return x, augmentation_tags
