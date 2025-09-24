python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --threshold 0.5 \
    --ood_eval_method ReAct

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 3.0 \
    --threshold 3.2 \
    --ood_eval_method ReAct

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --threshold 3.0 \
    --ood_eval_method ReAct  

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --threshold 1.0 \
    --ood_eval_method ReAct

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 1.0 \
    --threshold 2.6 \
    --ood_eval_method ReAct

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet18 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --threshold 4.1 \
    --ood_eval_method ReAct  