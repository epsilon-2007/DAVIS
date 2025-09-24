python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model mobilenetv2_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg \
 --threshold 1.0 \
 --ood_eval_method ReAct

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model mobilenetv2_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg+std \
 --std 0.5 \
 --threshold 2.4 \
 --ood_eval_method ReAct

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model mobilenetv2_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type max \
 --threshold 5.6 \
 --ood_eval_method ReAct