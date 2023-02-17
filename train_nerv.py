from __future__ import print_function

import argparse
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_nerv import CustomDataSet, Generator
from utils import *


def main():
    parser = argparse.ArgumentParser()

    # dataset parameters
    # 数据集的参数
    parser.add_argument('--vid',  default=[None], type=int,  nargs='+', help='video id list for training')
    parser.add_argument('--scale', type=int, default=1, help='scale-up facotr for data transformation,  added to suffix!!!!')
    parser.add_argument('--frame_gap', type=int, default=1, help='frame selection gap')
    parser.add_argument('--augment', type=int, default=0, help='augment frames between frames,  added to suffix!!!!')
    parser.add_argument('--dataset', type=str, default='UVG', help='dataset',)
    parser.add_argument('--test_gap', default=1, type=int, help='evaluation gap')

# NERV architecture parameters
    # embedding parameters
    # 位置编码的参数
    parser.add_argument('--embed', type=str, default='1.25_80', help='base value/embed length for position encoding')

    # FC + Conv parameters
    # 全连接层和卷积层的参数
    parser.add_argument('--stem_dim_num', type=str, default='1024_1', help='hidden dimension and length')
    parser.add_argument('--fc_hw_dim', type=str, default='9_16_128', help='out size (h,w) for mlp')
    parser.add_argument('--expansion', type=float, default=8, help='channel expansion from fc to conv')
    parser.add_argument('--reduction', type=int, default=2)
    parser.add_argument('--strides', type=int, nargs='+', default=[5, 3, 2, 2, 2], help='strides list')
    parser.add_argument('--num-blocks', type=int, default=1)

    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    parser.add_argument('--lower-width', type=int, default=32, help='lowest channel width for output feature maps')
    parser.add_argument("--single_res", action='store_true', help='single resolution,  added to suffix!!!!')
    parser.add_argument("--conv_type", default='conv', type=str,  help='upscale methods, can add bilinear and deconvolution methods', choices=['conv', 'deconv', 'bilinear'])

    # General training setups
    # 一般训练设置参数
    parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('-b', '--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--not_resume_epoch', action='store_true', help='resuming start_epoch from checkpoint')
    parser.add_argument('-e', '--epochs', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--cycles', type=int, default=1, help='epoch cycles for training')
    parser.add_argument('--warmup', type=float, default=0.2, help='warmup epoch ratio compared to the epochs, default=0.2,  added to suffix!!!!')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
    parser.add_argument('--lr_type', type=str, default='cosine', help='learning rate type, default=cosine')
    parser.add_argument('--lr_steps', default=[], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10,  added to suffix!!!!')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for adam. default=0.5,  added to suffix!!!!')
    parser.add_argument('--loss_type', type=str, default='L2', help='loss type, default=L2')
    parser.add_argument('--lw', type=float, default=1.0, help='loss weight,  added to suffix!!!!')
    parser.add_argument('--sigmoid', action='store_true', help='using sigmoid for output prediction')

    # evaluation parameters
    # 测试参数
    parser.add_argument('--eval_only', action='store_true', default=False, help='do evaluation only')
    parser.add_argument('--eval_freq', type=int, default=50, help='evaluation frequency,  added to suffix!!!!')
    parser.add_argument('--quant_bit', type=int, default=-1, help='bit length for model quantization')
    parser.add_argument('--quant_axis', type=int, default=0, help='quantization axis (-1 means per tensor)')
    parser.add_argument('--dump_images', action='store_true', default=False, help='dump the prediction images')
    parser.add_argument('--eval_fps', action='store_true', default=False, help='fwd multiple times to test the fps ')

    # pruning paramaters
    # 修建模型的参数
    parser.add_argument('--prune_steps', type=float, nargs='+', default=[0.,], help='prune steps')
    parser.add_argument('--prune_ratio', type=float, default=1.0, help='pruning ratio')

    # distribute learning parameters
    # 分布式学习参数
    parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:9888', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

    # logging, output directory, 
    # 保存日志和输出目录的参数
    parser.add_argument('--debug', action='store_true', help='defbug status, earlier for train/eval')  
    parser.add_argument('-p', '--print-freq', default=50, type=int,)
    parser.add_argument('--weight', default='None', type=str, help='pretrained weights for ininitialization')
    parser.add_argument('--overwrite', action='store_true', help='overwrite the output dir if already exists')
    parser.add_argument('--outf', default='unify', help='folder to output images and model checkpoints')
    parser.add_argument('--suffix', default='', help="suffix str for outf")

    # 参数
    args = parser.parse_args()
        
    # warm_up 参数
    args.warmup = int(args.warmup * args.epochs)

    # 把参数都打印出来
    print(args)
    # 设置浮点数的精确位数
    torch.set_printoptions(precision=4) 
    
    # 是否是debug模式，如果是debug模式就把eval_freq设成每次都评估
    # 并且将日志输出到output/debug模式
    if args.debug:
        args.eval_freq = 1
        args.outf = 'output/debug'
    else:
        # 不用debug模式就直接把输出目录join起来
        args.outf = os.path.join('output', args.outf)

    # 如果有修剪因子但是没有测试参数，则运行if
    if args.prune_ratio < 1 and not args.eval_only: 
        prune_str = '_Prune{}_{}'.format(args.prune_ratio, ','.join([str(x) for x in args.prune_steps]))
    else:
        prune_str = ''
    
    # 也是里面的一个字符串，不用太在意
    # 额外字符串，是显示是否是分布式，是否是测试模式等
    extra_str = '_Strd{}_{}Res{}{}'.format( ','.join([str(x) for x in args.strides]),  'Sin' if args.single_res else f'_lw{args.lw}_multi',  
            '_dist' if args.distributed else '', f'_eval' if args.eval_only else '')
    # 是否有归一化
    norm_str = '' if args.norm == 'none' else args.norm

    # 输出文件名，直接包含各种参数，看的更简单点
    exp_id = f'{args.dataset}/embed{args.embed}_{args.stem_dim_num}_fc_{args.fc_hw_dim}__exp{args.expansion}_reduce{args.reduction}_low{args.lower_width}_blk{args.num_blocks}_cycle{args.cycles}' + \
            f'_gap{args.frame_gap}_e{args.epochs}_warm{args.warmup}_b{args.batchSize}_{args.conv_type}_lr{args.lr}_{args.lr_type}' + \
            f'_{args.loss_type}{norm_str}{extra_str}{prune_str}'
    
    # 激活函数类型和后缀，但其实没有后缀
    exp_id += f'_act{args.act}_{args.suffix}'
    # 得到最终的输出文件名
    args.exp_id = exp_id
    
    # 拿到最终路径
    args.outf = os.path.join(args.outf, exp_id)
    
    # 是否覆盖之前的文件
    if args.overwrite and os.path.isdir(args.outf):
    	print('Will overwrite the existing output dir!')
    	shutil.rmtree(args.outf)

    # 如果不存在就创建路径
    if not os.path.isdir(args.outf):
        os.makedirs(args.outf)

    # 感觉没什么用
    # 没有看到这句话具体在哪使用了
    # 用于下面的分布式训练，但是本文没有用到
    port = hash(args.exp_id) % 20000 + 10000
    args.init_method =  f'tcp://127.0.0.1:{port}'
    print(f'init_method: {args.init_method}', flush=True)
    
    # 设置输出精度
    torch.set_printoptions(precision=2) 
    # 计算CUDA设备的个数，用于下方是否能使用分布式训练
    # 但其实也没用到
    args.ngpus_per_node = torch.cuda.device_count()
    
    # 分布式训练的东西，不需要管
    if args.distributed and args.ngpus_per_node > 1:
        mp.spawn(train, nprocs=args.ngpus_per_node, args=(args,))
    else:
        # 运行训练函数
        train(None, args)

def train(local_rank, args):
    # 随机种子使用了1，等会可以调整试试
    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    # 定义四个空向量，然后定义两个标志位
    train_best_psnr, train_best_msssim, val_best_psnr, val_best_msssim = [torch.tensor(0) for _ in range(4)]
    is_train_best, is_val_best = False, False

    # 生成位置编码对象，把参数先给进去
    PE = PositionalEncoding(args.embed)
    # 把编码长度给到一个参数
    args.embed_length = PE.embed_length
    # 各种参数赋传入生成器中，即NeRV主干
    model = Generator(embed_length=args.embed_length, stem_dim_num=args.stem_dim_num, fc_hw_dim=args.fc_hw_dim, expansion=args.expansion, 
        num_blocks=args.num_blocks, norm=args.norm, act=args.act, bias = True, reduction=args.reduction, conv_type=args.conv_type,
        stride_list=args.strides,  sin_res=args.single_res,  lower_width=args.lower_width, sigmoid=args.sigmoid)

    ##### prune model params and flops #####
    # 修剪因子小于1，则表示需要修剪
    prune_net = args.prune_ratio < 1
    # import pdb; pdb.set_trace; from IPython import embed; embed()
    # 需要修剪则
    if prune_net:
        # 定义一个空参数列表
        param_list = []
        # 
        for k,v in model.named_parameters():
            # 如果是权重
            if 'weight' in k:
                # 在全连接网络中，先split切分出来，然后加到参数列表中
                if 'stem' in k:
                    stem_ind = int(k.split('.')[1])
                    param_list.append(model.stem[stem_ind])
                # 在卷积层中，也是split切分出来然后加入参数列表
                elif 'layers' in k[:6] and 'conv' in k:
                    layer_ind = int(k.split('.')[1])
                    param_list.append(model.layers[layer_ind].conv.conv)
        # 需要修剪的参数列表
        # 全局非结构化剪枝的一个参数
        param_to_prune = [(ele, 'weight') for ele in param_list]
        # 基础修剪因子
        prune_base_ratio = args.prune_ratio ** (1. / len(args.prune_steps))
        args.prune_steps = [int(x * args.epochs) for x in args.prune_steps]
        prune_num = 0
        # 在测试情况下，对模型进行修剪
        if args.eval_only:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )

    ##### get model params and flops #####
    # 计算所有参数数量
    total_params = sum([p.data.nelement() for p in model.parameters()]) / 1e6
    
    # local_rank = 0
    if local_rank in [0, None]:
        # 也是计算所有参数数量
        params = sum([p.data.nelement() for p in model.parameters()]) / 1e6

        # 输出参数数量
        print(f'{args}\n {model}\n Model Params: {params}M')
        # 在rank0文本文件中写入模型信息和参数量信息
        with open('{}/rank0.txt'.format(args.outf), 'a') as f:
            f.write(str(model) + '\n' + f'Params: {params}M\n')
        # 创建一个tensorboard文件保存路径，文件名就叫param_xxxM
        writer = SummaryWriter(os.path.join(args.outf, f'param_{total_params}M', 'tensorboard'))
    else:
        writer = None

    # distrite model to gpu or parallel
    # 使用哪个GPU进行训练
    print("Use GPU: {} for training".format(local_rank))
    # 如果分布式则弄到其他卡上搞分布式训练
    if args.distributed and args.ngpus_per_node > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.ngpus_per_node,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()        
        args.batchSize = int(args.batchSize / args.ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.to(local_rank), device_ids=[local_rank], \
                                                          output_device=local_rank, find_unused_parameters=False)
    elif args.ngpus_per_node > 1:
        model = torch.nn.DataParallel(model).cuda() #model.cuda() #
    else:
        # 转到GPU
        model = model.cuda()

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), betas=(args.beta, 0.999))

    # resume from args.weight
    # 从指向的参数路径恢复
    checkpoint = None
    # 使用的GPU个数
    loc = 'cuda:{}'.format(local_rank if local_rank is not None else 0)
    # 这个路径不是空的
    if args.weight != 'None':
        # 将路径打印到屏幕
        print("=> loading checkpoint '{}'".format(args.weight))
        # 赋值-->加载参数--> orig_ckt 原始参数字典--> new_ckt 替换了一次
        checkpoint_path = args.weight
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        orig_ckt = checkpoint['state_dict']
        new_ckt={k.replace('blocks.0.',''):v for k,v in orig_ckt.items()}
        
        # 判断某些条件是否继续替换
        if 'module' in list(orig_ckt.keys())[0] and not hasattr(model, 'module'):
            new_ckt={k.replace('module.',''):v for k,v in new_ckt.items()}
            model.load_state_dict(new_ckt)
        elif 'module' not in list(orig_ckt.keys())[0] and hasattr(model, 'module'):
            model.module.load_state_dict(new_ckt)
        else:
            model.load_state_dict(new_ckt)
        # 打印加载权重路径和epoch数量
        print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, checkpoint['epoch']))        

    # resume from model_latest
    # 从最新的pth文件加载
    checkpoint_path = os.path.join(args.outf, 'model_latest.pth')
    # 有这个路径就加载
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        # 小于一就修剪
        if prune_net:
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num, 
            )
            
            sparisity_num = 0.
            # 计算稀疏参数的数量
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            # 打印稀疏比例
            print(f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}')
        # 加载模型
        model.load_state_dict(checkpoint['state_dict'])
        # 自动从某epoch加载，重新开始训练
        print("=> Auto resume loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
        print("=> No resume checkpoint found at '{}'".format(checkpoint_path))

    # 
    args.start_epoch = 0
    if checkpoint is not None:
        # 加载检查点
        args.start_epoch = checkpoint['epoch'] 
        train_best_psnr = checkpoint['train_best_psnr'].to(torch.device(loc))
        train_best_msssim = checkpoint['train_best_msssim'].to(torch.device(loc))
        val_best_psnr = checkpoint['val_best_psnr'].to(torch.device(loc))
        val_best_msssim = checkpoint['val_best_msssim'].to(torch.device(loc))
        optimizer.load_state_dict(checkpoint['optimizer'])

    if args.not_resume_epoch:
        args.start_epoch = 0

    # setup dataloader
    # 数据加载器
    img_transforms = transforms.ToTensor()
    DataSet = CustomDataSet
    train_data_dir = f'./data/{args.dataset.lower()}'
    val_data_dir = f'./data/{args.dataset.lower()}'

    # 训练集
    train_dataset = DataSet(train_data_dir, img_transforms,vid_list=args.vid, frame_gap=args.frame_gap,  )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=(train_sampler is None),
         num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, worker_init_fn=worker_init_fn)

    # 测试集 其实是一样的
    val_dataset = DataSet(val_data_dir, img_transforms, vid_list=args.vid, frame_gap=args.test_gap,  )
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if args.distributed else None
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize,  shuffle=False,
         num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False, worker_init_fn=worker_init_fn)
    # 数据集长度
    data_size = len(train_dataset)

    # 测试情况下，添加--eval_Only命令
    if args.eval_only:
        print('Evaluation ...')
        # 设定时间帧
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        # 设置打印字段
        print_str = f'{time_str}\t Results for checkpoint: {args.weight}\n'
        # 修剪并打印稀疏系数
        if prune_net:
            for param in param_to_prune:
                prune.remove(param[0], param[1])
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print_str += f'Model sparsity at Epoch{args.start_epoch}: {sparisity_num / 1e6 / total_params}\n'

        # import pdb; pdb.set_trace; from IPython import embed; embed()
        val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)
        print_str += f'PSNR/ms_ssim on validate set for bit {args.quant_bit} with axis {args.quant_axis}: {round(val_psnr.item(),2)}/{round(val_msssim.item(),4)}'
        print(print_str)
        # 创建一个文本文件写入信息
        with open('{}/eval.txt'.format(args.outf), 'a') as f:
            f.write(print_str + '\n\n')        
        return

    # Training
    start = datetime.now()
    total_epochs = args.epochs * args.cycles
    
    # 训练
    for epoch in range(args.start_epoch, total_epochs):
        model.train()
        ##### prune the network if needed #####
        # 修剪模型
        if prune_net and epoch in args.prune_steps:
            prune_num += 1 
            prune.global_unstructured(
                param_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=1 - prune_base_ratio ** prune_num,
            )
            
            sparisity_num = 0.
            for param in param_list:
                sparisity_num += (param.weight == 0).sum()
            print(f'Model sparsity at Epoch{epoch}: {sparisity_num / 1e6 / total_params}')
        # 设置时间和空列表
        epoch_start_time = datetime.now()
        psnr_list = []
        msssim_list = []
        # iterate over dataloader
        # 整个数据集循环
        for i, (data,  norm_idx) in enumerate(train_dataloader):
            # 该部分防止出错
            if i > 10 and args.debug:
                break
            # 位置编码
            embed_input = PE(norm_idx)
            
            # 默认是None，没啥差别
            if local_rank is not None:
                data = data.cuda(local_rank, non_blocking=True)
                embed_input = embed_input.cuda(local_rank, non_blocking=True)
            else:
                # non_blocking保证数据并行传输
                data,  embed_input = data.cuda(non_blocking=True),   embed_input.cuda(non_blocking=True)
            # forward and backward
            # 输入模型并得到输出
            output_list = model(embed_input)
            # 弄到一样的大小
            target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
            # 计算损失
            loss_list = [loss_fn(output, target, args) for output, target in zip(output_list, target_list)]
            loss_list = [loss_list[i] * (args.lw if i < len(loss_list) - 1 else 1) for i in range(len(loss_list))]
            loss_sum = sum(loss_list)
            
            # 调整学习率
            lr = adjust_lr(optimizer, epoch % args.epochs, i, data_size, args)
            
            # 反向传播过程
            optimizer.zero_grad()
            loss_sum.backward()
            optimizer.step()

            # compute psnr and msssim
            # 计算MSSSIM
            psnr_list.append(psnr_fn(output_list, target_list))
            msssim_list.append(msssim_fn(output_list, target_list))
            if i % args.print_freq == 0 or i == len(train_dataloader) - 1:
                train_psnr = torch.cat(psnr_list, dim=0) #(batchsize, num_stage)
                train_psnr = torch.mean(train_psnr, dim=0) #(num_stage)
                train_msssim = torch.cat(msssim_list, dim=0) #(batchsize, num_stage)
                train_msssim = torch.mean(train_msssim.float(), dim=0) #(num_stage)
                time_now_string = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
                print_str = '[{}] Rank:{}, Epoch[{}/{}], Step [{}/{}], lr:{:.2e} PSNR: {}, MSSSIM: {}'.format(
                    time_now_string, local_rank, epoch+1, args.epochs, i+1, len(train_dataloader), lr, 
                    RoundTensor(train_psnr, 2, False), RoundTensor(train_msssim, 4, False))
                print(print_str, flush=True)
                if local_rank in [0, None]:
                    with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                        f.write(print_str + '\n')

        # collect numbers from other gpus
        # 没有用分布式，这一部分也不会用到
        if args.distributed and args.ngpus_per_node > 1:
            train_psnr = all_reduce([train_psnr.to(local_rank)])
            train_msssim = all_reduce([train_msssim.to(local_rank)])

        # ADD train_PSNR TO TENSORBOARD
        # 将训练PSNR添加到tensorboard文件中
        if local_rank in [0, None]:
            h, w = output_list[-1].shape[-2:]
            is_train_best = train_psnr[-1] > train_best_psnr
            train_best_psnr = train_psnr[-1] if train_psnr[-1] > train_best_psnr else train_best_psnr
            train_best_msssim = train_msssim[-1] if train_msssim[-1] > train_best_msssim else train_best_msssim
            writer.add_scalar(f'Train/PSNR_{h}X{w}_gap{args.frame_gap}', train_psnr[-1].item(), epoch+1)
            writer.add_scalar(f'Train/MSSSIM_{h}X{w}_gap{args.frame_gap}', train_msssim[-1].item(), epoch+1)
            writer.add_scalar(f'Train/best_PSNR_{h}X{w}_gap{args.frame_gap}', train_best_psnr.item(), epoch+1)
            writer.add_scalar(f'Train/best_MSSSIM_{h}X{w}_gap{args.frame_gap}', train_best_msssim, epoch+1)
            print_str = '\t{}p: current: {:.2f}\t best: {:.2f}\t msssim_best: {:.4f}\t'.format(h, train_psnr[-1].item(), train_best_psnr.item(), train_best_msssim.item())
            print(print_str, flush=True)
            # 写入到rank0.txt文件中
            with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')
            writer.add_scalar('Train/lr', lr, epoch+1)
            epoch_end_time = datetime.now()
            print("Time/epoch: \tCurrent:{:.2f} \tAverage:{:.2f}".format( (epoch_end_time - epoch_start_time).total_seconds(), \
                    (epoch_end_time - start).total_seconds() / (epoch + 1 - args.start_epoch) ))

        # 保存检查点文件
        state_dict = model.state_dict()
        save_checkpoint = {
            'epoch': epoch+1,
            'state_dict': state_dict,
            'train_best_psnr': train_best_psnr,
            'train_best_msssim': train_best_msssim,
            'val_best_psnr': val_best_psnr,
            'val_best_msssim': val_best_msssim,
            'optimizer': optimizer.state_dict(),   
        }    
        # evaluation
        # 测试，每五十次检查一次并且最后十次每次都检查
        if (epoch + 1) % args.eval_freq == 0 or epoch > total_epochs - 10:
            val_start_time = datetime.now()
            val_psnr, val_msssim = evaluate(model, val_dataloader, PE, local_rank, args)
            val_end_time = datetime.now()
            if args.distributed and args.ngpus_per_node > 1:
                val_psnr = all_reduce([val_psnr.to(local_rank)])
                val_msssim = all_reduce([val_msssim.to(local_rank)])            
            if local_rank in [0, None]:
                # ADD val_PSNR TO TENSORBOARD
                h, w = output_list[-1].shape[-2:]
                print_str = f'Eval best_PSNR at epoch{epoch+1}:'
                is_val_best = val_psnr[-1] > val_best_psnr
                val_best_psnr = val_psnr[-1] if is_val_best else val_best_psnr
                val_best_msssim = val_msssim[-1] if val_msssim[-1] > val_best_msssim else val_best_msssim
                writer.add_scalar(f'Val/PSNR_{h}X{w}_gap{args.test_gap}', val_psnr[-1], epoch+1)
                writer.add_scalar(f'Val/MSSSIM_{h}X{w}_gap{args.test_gap}', val_msssim[-1], epoch+1)
                writer.add_scalar(f'Val/best_PSNR_{h}X{w}_gap{args.test_gap}', val_best_psnr, epoch+1)
                writer.add_scalar(f'Val/best_MSSSIM_{h}X{w}_gap{args.test_gap}', val_best_msssim, epoch+1)
                print_str += '\t{}p: current: {:.2f}\tbest: {:.2f} \tbest_msssim: {:.4f}\t Time/epoch: {:.2f}'.format(h, val_psnr[-1].item(),
                     val_best_psnr.item(), val_best_msssim.item(), (val_end_time - val_start_time).total_seconds())
                print(print_str)
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
                if is_val_best:
                    torch.save(save_checkpoint, '{}/model_val_best.pth'.format(args.outf))

        if local_rank in [0, None]:
            # state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(save_checkpoint, '{}/model_latest.pth'.format(args.outf))
            if is_train_best:
                torch.save(save_checkpoint, '{}/model_train_best.pth'.format(args.outf))

    print("Training complete in: " + str(datetime.now() - start))


@torch.no_grad()
def evaluate(model, val_dataloader, pe, local_rank, args):
    # Model Quantization
    # 不等于-1就要量化了
    if args.quant_bit != -1:
        # 现有ckt字典
        cur_ckt = model.state_dict()
        # 导入霍夫曼编码
        from dahuffman import HuffmanCodec
        # 量化权重表
        quant_weitht_list = []
        # 
        for k,v in cur_ckt.items():
            # 一个判断，目前还搞不懂干啥的
            large_tf = (v.dim() in {2,4} and 'bias' not in k)
            # 量化每个向量
            quant_v, new_v = quantize_per_tensor(v, args.quant_bit, args.quant_axis if large_tf else -1)
            # 只包含非零项
            valid_quant_v = quant_v[v!=0] # only include non-zero weights
            # 
            quant_weitht_list.append(valid_quant_v.flatten())
            cur_ckt[k] = new_v
        cat_param = torch.cat(quant_weitht_list)
        input_code_list = cat_param.tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        # generating HuffmanCoding table
        # 生成码表
        codec = HuffmanCodec.from_data(input_code_list)
        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        # 计算平均码长
        avg_bits = total_bits / len(input_code_list)    
        # import pdb; pdb.set_trace; from IPython import embed; embed()  
        # 打印熵编码效率     
        encoding_efficiency = avg_bits / args.quant_bit
        print_str = f'Entropy encoding efficiency for bit {args.quant_bit}: {encoding_efficiency}'
        print(print_str)

        # 记录到eval.txt文件中
        if local_rank in [0, None]:
            with open('{}/eval.txt'.format(args.outf), 'a') as f:
                f.write(print_str + '\n')       
        model.load_state_dict(cur_ckt)

        # import pdb; pdb.set_trace; from IPython import embed; embed()

    psnr_list = []
    msssim_list = []
    
    # 如果需要可视化，就保存图像在visualize文件中
    if args.dump_images:
        # 导入保存图像函数
        from torchvision.utils import save_image
        visual_dir = f'{args.outf}/visualize'
        print(f'Saving predictions to {visual_dir}')
        # 创建保存路径
        if not os.path.isdir(visual_dir):
            os.makedirs(visual_dir)

    time_list = []
    model.eval()
    for i, (data,  norm_idx) in enumerate(val_dataloader):
        # 防出错
        if i > 10 and args.debug:
            break
        # 位置编码
        embed_input = pe(norm_idx)
        # CUDA
        if local_rank is not None:
            data = data.cuda(local_rank, non_blocking=True)
            embed_input = embed_input.cuda(local_rank, non_blocking=True)
        else:
            data,  embed_input = data.cuda(non_blocking=True), embed_input.cuda(non_blocking=True)

        # compute psnr and msssim
        # 计算PSNR和SSIM
        fwd_num = 10 if args.eval_fps else 1
        for _ in range(fwd_num):
            # embed_input = embed_input.half()
            # model = model.half()
            start_time = datetime.now()
            output_list = model(embed_input)
            torch.cuda.synchronize()
            # torch.cuda.current_stream().synchronize()
            time_list.append((datetime.now() - start_time).total_seconds())

        # dump predictions
        # 如果需要转存预测结果
        if args.dump_images:
            for batch_ind in range(args.batchSize):
                # 对于bunny数据集，是0-131共132张图像
                full_ind = i * args.batchSize + batch_ind
                # 将原始数据和预测结果都保存了
                save_image(output_list[-1][batch_ind], f'{visual_dir}/pred_{full_ind}.png')
                save_image(data[batch_ind], f'{visual_dir}/gt_{full_ind}.png')

        # compute psnr and ms-ssim
        # 计算PSNR和MS-SSIM
        target_list = [F.adaptive_avg_pool2d(data, x.shape[-2:]) for x in output_list]
        psnr_list.append(psnr_fn(output_list, target_list))
        msssim_list.append(msssim_fn(output_list, target_list))
        val_psnr = torch.cat(psnr_list, dim=0)              #(batchsize, num_stage)
        val_psnr = torch.mean(val_psnr, dim=0)              #(num_stage)
        val_msssim = torch.cat(msssim_list, dim=0)          #(batchsize, num_stage)
        val_msssim = torch.mean(val_msssim.float(), dim=0)  #(num_stage)        
        if i % args.print_freq == 0:
            fps = fwd_num * (i+1) * args.batchSize / sum(time_list)
            print_str = 'Rank:{}, Step [{}/{}], PSNR: {}, MSSSIM: {} FPS: {}'.format(
                local_rank, i+1, len(val_dataloader),
                RoundTensor(val_psnr, 2, False), RoundTensor(val_msssim, 4, False), round(fps, 2))
            print(print_str)
            if local_rank in [0, None]:
                with open('{}/rank0.txt'.format(args.outf), 'a') as f:
                    f.write(print_str + '\n')
    model.train()

    return val_psnr, val_msssim


if __name__ == '__main__':
    main()
