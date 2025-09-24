python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --ash_p 80 \
    --ood_eval_method ASH

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 3.0 \
    --ash_p 80 \
    --ood_eval_method ASH

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --ash_p 80 \
    --ood_eval_method ASH  

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --ash_p 80 \
    --ood_eval_method ASH

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 1.0 \
    --ash_p 80 \
    --ood_eval_method ASH

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model resnet34 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --ash_p 80 \
    --ood_eval_method ASH  