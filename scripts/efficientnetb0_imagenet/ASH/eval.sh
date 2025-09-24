python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model efficientnetb0_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg \
 --ash_p 70 \
 --ood_eval_method ASH

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model efficientnetb0_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg+std \
 --std 4.0 \
 --ash_p 50 \
 --ood_eval_method ASH

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model efficientnetb0_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type max \
 --ash_p 50 \
 --ood_eval_method ASH