python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model resnet50_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg \
 --threshold 1.0 \
 --ood_eval_method ReAct

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model resnet50_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg+std \
 --std 0.5 \
 --threshold 1.9 \
 --ood_eval_method ReAct

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model resnet50_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type max \
 --threshold 4.7 \
 --ood_eval_method ReAct