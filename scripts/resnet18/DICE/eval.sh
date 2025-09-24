python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --p 70 \
    --ood_eval_method DICE

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 3.0 \
    --p 70 \
    --ood_eval_method DICE

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --p 70 \
    --ood_eval_method DICE  

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --p 70 \
    --ood_eval_method DICE

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 1.0 \
    --p 70 \
    --ood_eval_method DICE

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --p 70 \
    --ood_eval_method DICE  