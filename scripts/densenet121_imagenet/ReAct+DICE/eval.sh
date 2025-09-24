python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg \
 --p 10 \
 --threshold 1.4 \
 --ood_eval_method ReAct+DICE

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg+std \
 --std 0.5 \
 --p 10 \
 --threshold 4.5 \
 --ood_eval_method ReAct+DICE

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type max \
 --p 10 \
 --threshold 11.0 \
 --ood_eval_method ReAct+DICE