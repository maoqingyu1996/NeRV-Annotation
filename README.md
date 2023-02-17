# NeRV: Neural Representations for Videos  (NeurIPS 2021)
### [Project Page](https://haochen-rye.github.io/NeRV) | [Paper](https://arxiv.org/abs/2110.13903) | [UVG Data](http://ultravideo.fi/#testsequences) 


[Hao Chen](https://haochen-rye.github.io),
Bo He,
Hanyu Wang,
Yixuan Ren,
Ser-Nam Lim],
[Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav/)<br>
This is the official implementation of the paper "NeRV: Neural Representations for Videos ".

## Method overview
<img src="https://i.imgur.com/OTdHe6r.png" width="560"  />

## Get started
我们在Python 3.8环境下运行，你可以通过以下命令设置一个conda环境并包含所有依赖：
```
pip install -r requirements.txt 
```

## High-Level structure
The code is organized as follows:
* [train_nerv.py](./train_nerv.py) 包含一个一般的训练程序
* [model_nerv.py](./model_nerv.py) 包含训练加载器和神经网络架构
* [data/](./data) 视频/图像数据目录，我们提供了Big Buck Bunny数据集在其中
* [checkpoints/](./checkpoints) 目录下包含了一些在Big Buck Bunny数据集上的预训练模型
* log files (tensorboard, txt, state_dict etc.) will be saved in output directory (specified by ```--outf```)
* 日志文件（Tensorboard文件，txt文件，参数字典等）将会保存在输出目录中（由--outf参数确定）

## Reproducing experiments 实验复现

### Training experiments 训练实验

NeRV-S在大巴兔数据集上的实验可以用下方的命令实现，使用```9_16_58``` 和 ```9_16_112```分别替换```fc_hw_dim```可以得到NeRV-M和NeRV-L的实验结果。
```
python train_nerv.py -e 300   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --act swish 
```

### Evaluation experiments 测试实验

为评估预训练模型，只需要添加 -- eval_Only 和通过 --weight 指定模型路径即可，你可以使用```--quant_bit [bit_lenght]```指定模型量化位宽，可以使用```--eval_fps``` 测试解码速度，我们在下方提供了NeRV在Bunny数据集上的命令样例。

```
python train_nerv.py -e 300   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none  --act swish \
    --weight checkpoints/nerv_S.pth --eval_only 
```

### Decoding: Dump predictions with pre-trained model  解码：使用预训练模型转存预测结果

为使用预训练模型转存预测结果，只需要添加```--dump_images``` 并且配上```--eval_Only```和指定模型路径```--weight```

```
python train_nerv.py -e 300   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none  --act swish \
   --weight checkpoints/nerv_S.pth --eval_only  --dump_images
```

## Model Pruning 模型剪枝

### Evaluate the pruned model 模型剪枝命令
修剪一个预训练模型和微调以恢复其性能，通过```--prune_ratio```指定模型参数修剪规模，```--weight```用于指定预训练模型，```--not_resume_epoch``` 用于跳过加载预训练权重循环以重新启动微调。

```
python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf prune_ab --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
    --weight checkpoints/nerv_S.pth --not_resume_epoch --prune_ratio 0.4 
```

### Evaluate the pruned model
To evaluate pruned model, using ```--weight``` to specify the pruned model weight, ```--prune_ratio``` to initialize the ```weight_mask``` for checkpoint loading, ```eval_only``` for evaluation mode, ```--quant_bit``` to specify quantization bit length, ```--quant_axis``` to specify quantization axis

为测试剪枝后模型，使用```--weight```指定剪枝后的模型权重，使用```--prune_ratio```初始化```weight_mask```用于检查点加载，```eval_only``` 用于测试模式，```--quant_bit```用于指定量化位长度，```--quant_axis```用于指定量化轴。
```
python train_nerv.py -e 100   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf dbg --embed 1.25_40 --stem_dim_num 512_1  --reduction 2  --fc_hw_dim 9_16_26 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0. --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --suffix 107  --act swish \
    --weight checkpoints/nerv_S_pruned.pth --prune_ratio 0.4  --eval_only --quant_bit 8 --quant_axis 0

```

### Distrotion-Compression result
The final bits-per-pixel (bpp) is computed by $$Model\_Parameter * (1 - Prune\_Ratio) * Quant\_Bit / Pixel\_Num$$.
We provide numerical results for distortion-compression (Figure 7, 8 and 11) at [psnr_bpp_results.csv](./checkpoints/psnr_bpp_results.csv) .

最终的bbp是通过以上公式计算的，我们提供了率失真压缩的数值结果，体现在正文的图7，8和11中。

## Citation
If you find our work useful in your research, please cite:
```
@inproceedings{hao2021nerv,
    author = {Hao Chen, Bo He, Hanyu Wang, Yixuan Ren, Ser-Nam Lim, Abhinav Shrivastava },
    title = {NeRV: Neural Representations for Videos s},
    booktitle = {NeurIPS},
    year={2021}
}
```

## Contact
If you have any questions, please feel free to email the authors.
