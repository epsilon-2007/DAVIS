python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg \
 --p 70 \
 --ood_eval_method DICE

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type avg+std \
 --p 70 \
 --std 0.5 \
 --ood_eval_method DICE  

python3 eval_ood.py \
 --score energy \
 --batch-size 64 \
 --model densenet121_imagenet \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_loc datasets/ood-imagenet/ \
 --ood_eval_type max \
 --p 70 \
 --ood_eval_method DICE 