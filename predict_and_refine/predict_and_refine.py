import torch
import torch.nn as nn
import math
import numpy
from torch.optim import Adam

def predict_and_refine(image, mask,inpainted_low_res, model, lr=0.001, n_iters=15):
    # 经过编码器的中间特征被定义为Z
    # 1000*1000
    z = model.front.forward(image, mask)
    # configure optimizer to update the feature map
    # 设置优化器，将中间特征Z置为优化器优化对象
    optimizer = Adam([z],lr)
    # 循环N次
    for _ in range(n_iters):
        # 优化器梯度设置为0
        optimizer.zero_grad()
        # 将中间特征Z输入解码器得到修复图像
        # 1000*1000
        inpainted = model.rear.forward(z)
        # 将修复图像进行下采样
        # 100*100
        inpainted_downscaled = downscale(inpainted)
        # 下采样的修复图像与
        # 100*100
        loss = l1_over_masked_region(
            inpainted_downscaled, inpainted_low_res, mask
        )
        # 反向传播与更新
        loss.backward()
        optimizer.step()
        # final forward pass
    return inpainted

def multiscale_inpaint(image, mask, model, smallest_scale=512):
    images, masks = build_pyramid(image, mask, smallest_scale)
    n_scales = len(images)
    # initialize with the lowest scale inpainting
    # 初始化最低尺度的图像修复
    inpainted = model.forward(images[0], masks[0])
    # 多个尺度循环
    for i in range(1, n_scales):
        # 多个尺度的images和mask
        image, mask = images[i],masks[i]
        # 
        inpainted_low_res = inpainted
        # 多个尺度优化
        inpainted = predict_and_refine(
            image, mask, inpainted_low_res, model
        )
    return inpainted