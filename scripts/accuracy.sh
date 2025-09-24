python3 accuracy.py \
 --batch-size 64 \
 --model mobilenetv2_imagenet  \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_eval_type avg \
 --ood_eval_method baseline/   

python3 accuracy.py \
 --batch-size 64 \
 --model mobilenetv2_imagenet  \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_eval_type avg+std \
 --std 2.0 \
 --ood_eval_method baseline/   

python3 accuracy.py \
 --batch-size 64 \
 --model mobilenetv2_imagenet  \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_eval_type max \
 --ood_eval_method baseline/  

python3 accuracy.py \
 --batch-size 64 \
 --model mobilenetv2_imagenet  \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_eval_type avg+std \
 --std 0.5 \
 --p 5 \
 --threshold 0.9 \
 --ood_eval_method ReAct+DICE   

python3 accuracy.py \
 --batch-size 64 \
 --model mobilenetv2_imagenet  \
 --id_loc datasets/in-imagenet/val \
 --in-dataset ImageNet-1K \
 --ood_eval_type max \
 --p 5 \
 --threshold 2.7 \
 --ood_eval_method ReAct+DICE   
