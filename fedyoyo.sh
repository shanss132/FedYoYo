device_id=2
noniid=0.5
imb_factor=0.1
dst='cifar10'
arch="resnet8"
method="fedyoyo"
num_rounds=300
lamda=4.0
gamma=0.1
warmup=50

CUDA_VISIBLE_DEVICES=$device_id python -u main_fedyoyo_github.py \
    --noniid $noniid \
    --imb_factor $imb_factor \
    --dst $dst \
    --num_rounds $num_rounds \
    --arch $arch \
    --method $method \
    --gamma $gamma \
    --warmup $warmup \
    --lamda $lamda \

#  nohup bash fedyoyo_github.sh 
