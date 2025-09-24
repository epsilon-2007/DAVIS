python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --threshold 0.5 \
    --ood_eval_method ReAct

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 3.0 \
    --threshold 3.4 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --threshold 3.1 \
    --ood_eval_method ReAct  

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --threshold 1.0 \
    --ood_eval_method ReAct


python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 1.0 \
    --threshold 2.8 \
    --ood_eval_method ReAct 

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --threshold 4.2 \
    --ood_eval_method ReAct  