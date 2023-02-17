CUDA_VISIBLE_DEVICES=2  python train_nerv.py -e 300   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_l_128 --embed 1.25_40 --stem_dim_num 128_1  --reduction 2  --fc_hw_dim 9_16_112 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --act swish
CUDA_VISIBLE_DEVICES=2  python train_nerv.py -e 300   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_l_256 --embed 1.25_40 --stem_dim_num 256_1  --reduction 2  --fc_hw_dim 9_16_112 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --act swish
CUDA_VISIBLE_DEVICES=2  python train_nerv.py -e 300   --lower-width 96 --num-blocks 1 --dataset bunny --frame_gap 1 \
    --outf bunny_l_384 --embed 1.25_40 --stem_dim_num 384_1  --reduction 2  --fc_hw_dim 9_16_112 --expansion 1  \
    --single_res --loss Fusion6   --warmup 0.2 --lr_type cosine  --strides 5 2 2 2 2  --conv_type conv \
    -b 1  --lr 0.0005 --norm none --act swish