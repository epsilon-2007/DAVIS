python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --threshold 1.2 \
    --ood_eval_method ReAct

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 1.0 \
    --threshold 2.9 \
    --ood_eval_method ReAct

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --threshold 6.5 \
    --ood_eval_method ReAct  

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --threshold 1.8 \
    --ood_eval_method ReAct

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 3.0 \
    --threshold 9.5 \
    --ood_eval_method ReAct

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model densenet101 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --threshold 10.0 \
    --ood_eval_method ReAct    