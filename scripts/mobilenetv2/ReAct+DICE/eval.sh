python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --p 70 \
    --threshold 0.5 \
    --ood_eval_method ReAct+DICE

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --std 3.0 \
    --ood_eval_type avg+std \
    --p 70 \
    --threshold 1.9 \
    --ood_eval_method ReAct+DICE

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --p 70 \
    --threshold 1.4 \
    --ood_eval_method ReAct+DICE

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --p 70 \
    --threshold 0.6 \
    --ood_eval_method ReAct+DICE

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 1.0 \
    --p 70 \
    --threshold 1.1 \
    --ood_eval_method ReAct+DICE

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --p 70 \
    --threshold 2.7 \
    --ood_eval_method ReAct+DICE