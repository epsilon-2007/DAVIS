python3 eval_ood.py \
    --score odin \
    --temp 1000 \
    --noise 0.004 \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --ood_eval_method baseline/odin  

python3 eval_ood.py \
    --score odin \
    --temp 1000 \
    --noise 0.004 \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --std 3.0 \
    --ood_eval_type avg+std \
    --ood_eval_method baseline/odin

python3 eval_ood.py \
    --score odin \
    --temp 1000 \
    --noise 0.004 \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --ood_eval_method baseline/odin   

python3 eval_ood.py \
    --score odin \
    --temp 1000 \
    --noise 0.004 \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --ood_eval_method baseline/odin  

python3 eval_ood.py \
    --score odin \
    --temp 1000 \
    --noise 0.004 \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --std 3.0 \
    --ood_eval_type avg+std \
    --ood_eval_method baseline/odin

python3 eval_ood.py \
    --score odin \
    --temp 1000 \
    --noise 0.004 \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --ood_eval_method baseline/odin   