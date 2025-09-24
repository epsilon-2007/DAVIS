python3 eval_ood.py \
 --score odin \
 --temp 1000 \
 --noise 0.0015 \
 --batch-size 64 \
 --model efficientnetb0_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg \
 --ood_eval_method baseline/odin

python3 eval_ood.py \
 --score odin \
 --temp 1000 \
 --noise 0.0015 \
 --batch-size 64 \
 --model efficientnetb0_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg+std \
 --std 0.5 \
 --ood_eval_method baseline/odin

python3 eval_ood.py \
 --score odin \
 --temp 1000 \
 --noise 0.0015 \
 --batch-size 64 \
 --model efficientnetb0_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type max \
 --ood_eval_method baseline/odin