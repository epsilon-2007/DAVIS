python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --ash_p 70 \
    --ood_eval_method ASH

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --std 3.0 \
    --ood_eval_type avg+std \
    --ash_p 70 \
    --ood_eval_method ASH

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-10 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --ash_p 70 \
    --ood_eval_method ASH

    
python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg \
    --ash_p 90 \
    --ood_eval_method ASH

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type avg+std \
    --std 3.0 \
    --ash_p 85 \
    --ood_eval_method ASH

python3 eval_ood.py \
    --score energy \
    --batch-size 64 \
    --model mobilenetv2 \
    --id_loc datasets/in/ \
    --in-dataset CIFAR-100 \
    --ood_loc datasets/ood/ \
    --ood_eval_type max \
    --ash_p 85 \
    --ood_eval_method ASH