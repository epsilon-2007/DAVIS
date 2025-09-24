python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model efficientnetb0_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg \
 --threshold 0.9 \
 --ood_eval_method ReAct

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model efficientnetb0_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg+std \
 --std 4.0 \
 --threshold 7.7 \
 --ood_eval_method ReAct

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model efficientnetb0_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type max \
 --threshold 7.0 \
 --ood_eval_method ReAct