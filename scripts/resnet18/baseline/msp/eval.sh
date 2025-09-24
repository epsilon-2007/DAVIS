python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --ood_eval_method baseline/msp  

python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --std 1.0 \
    --ood_eval_type avg+std \
    --ood_eval_method baseline/msp

python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --ood_eval_method baseline/msp    

python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --ood_eval_method baseline/msp  

python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --std 1.0 \
    --ood_eval_type avg+std \
    --ood_eval_method baseline/msp

python3 eval_ood.py \
    --score msp \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --ood_eval_method baseline/msp    