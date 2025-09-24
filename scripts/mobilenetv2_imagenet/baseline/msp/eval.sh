python3 eval_ood.py \
 --score msp \
 --batch-size 64 \
 --model mobilenetv2_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg \
 --ood_eval_method baseline/msp

python3 eval_ood.py \
 --score msp \
 --batch-size 64 \
 --model mobilenetv2_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg+std \
 --std 0.5 \
 --ood_eval_method baseline/msp

python3 eval_ood.py \
 --score msp \
 --batch-size 64 \
 --model mobilenetv2_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type max \
 --ood_eval_method baseline/msp