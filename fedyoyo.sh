device_id=2
noniid=0.5
imb_factor=0.01
dst='cifar10'
arch="resnet8"
method="fedyoyo"
num_rounds=300
lamda=4.0
gamma=0.1
warmup=50

CUDA_VISIBLE_DEVICES=$device_id python -u main_fedyoyo.py \
    --noniid $noniid \
    --imb_factor $imb_factor \
    --dst $dst \
    --num_rounds $num_rounds \
    --arch $arch \
    --method $method \
    --gamma $gamma \
    --warmup $warmup \
    --lamda $lamda \

#  nohup bash fedyoyo.sh > fedyoyo_cifar10_noniid0.5_imb0.01.log 2>&1 &
