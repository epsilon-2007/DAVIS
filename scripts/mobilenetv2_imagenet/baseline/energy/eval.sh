python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model mobilenetv2_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg \
 --ood_eval_method baseline/energy

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model mobilenetv2_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg+std \
 --std 2.0 \
 --ood_eval_method baseline/energy

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model mobilenetv2_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type max \
 --ood_eval_method baseline/energy