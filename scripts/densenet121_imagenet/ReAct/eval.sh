python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg \
 --threshold 1.4 \
 --ood_eval_method ReAct

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg+std \
 --threshold 4.5 \
 --std 0.5 \
 --ood_eval_method ReAct

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type max \
 --threshold 11.0 \
 --ood_eval_method ReAct