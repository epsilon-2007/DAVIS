import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def ash_s(x, p):
    # x := [bt, d]
    batch_size, c = x.shape

    s1 = x.sum(dim=1) 
    k = c - int(np.round(c * p / 100.0))
    v, i = torch.topk(x, k, dim=1)
    x_pruned = torch.zeros_like(x)
    x_pruned.scatter_(dim=1, index=i, src=v)

    s2 = x_pruned.sum(dim=1)
    scale = s1 / (s2 ) #+ 1e-8)
    x_sharpened = x_pruned * torch.exp(scale[:, None])

    return x_sharpened

def scale_features(x, p):
    batch_size, c = x.shape
    thresh = torch.quantile(x, p / 100.0, dim=1, keepdim=True, interpolation='higher')

    sum_all = x.sum(dim=1, keepdim=True)  # [batch_size, 1]
    mask = x >= thresh  # boolean mask, [batch_size, D]

    sum_top = (x * mask).sum(dim=1, keepdim=True)  # [batch_size, 1]
    sum_top = torch.clamp(sum_top, min=1e-6)
    
    r = sum_all / sum_top  # [batch_size, 1]
    r = torch.exp(r)
    return r * x


def get_features(x, dim_in, args):
    # Rectified Activation
    if args.threshold is not None:
        x = x.clip(max = args.threshold) 
    # ASH
    if args.ash_p is not None:
        x = x.view(-1, dim_in)
        x = ash_s(x, args.ash_p)
    # SCALE
    if args.scale_p is not None:
        x = x.view(-1, dim_in)
        x = scale_features(x, args.scale_p)
    return x

def get_logits(inputs, model, args, logits=None):
    if logits is None:
        model.eval()
        with torch.inference_mode():
            if args.in_dataset == 'ImageNet-1K':
                if args.model in ['mobilenetv2_imagenet', 'densenet121_imagenet', 'efficientnetb0_imagenet']:
                    logits = model.classifier(inputs)
                elif args.model in ['swinB_imagenet', 'swinT_imagenet']:
                    logits = model.head(inputs)
                else:
                    logits = model.fc(inputs)
            elif args.in_dataset in ["CIFAR-10", "CIFAR-100"]:
                logits = model.output_layer(inputs)     
    return  logits

def get_msp_score(inputs, model, args, logits=None):
    x = inputs
    x = get_features(x, model.dim_in, args)
    logits = get_logits(x, model, args)
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores

def get_energy_score(inputs, model, args, logits=None):
    x = inputs
    x = get_features(x, model.dim_in, args)
    logits = get_logits(x, model, args)
    # temperature scaling
    logits = logits / args.temp 
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores

def get_odin_feature(x, model, args):
    avgpool = nn.AdaptiveAvgPool2d((1,1))
    maxpool = nn.AdaptiveMaxPool2d((1,1))

    if args.ood_eval_type == 'avg':
        features = model.my_features(x)
    elif args.ood_eval_type == 'avg+std':
        activation_map = model.my_encoder(x)
        avg =  avgpool(activation_map)
        std = activation_map.std(dim = (2, 3))
        avg = avg.view(-1, model.dim_in)
        std = std.view(-1, model.dim_in)
        features = avg + (args.std * std)
    elif args.ood_eval_type == 'max':  
            activation_map = model.my_encoder(x)
            features = maxpool(activation_map)
            features = features.view(-1, model.dim_in)
    return features

def get_odin_logit(inputs, model, args):
    if args.in_dataset == 'ImageNet-1K':
        if args.model in ['mobilenetv2_imagenet', 'densenet121_imagenet', 'efficientnetb0_imagenet']:
            logits = model.classifier(inputs)
        elif args.model in ['swinB_imagenet', 'swinT_imagenet']:
            logits = model.head(inputs)
        else:
            logits = model.fc(inputs)
    elif args.in_dataset in ["CIFAR-10", "CIFAR-100"]:
        logits = model.output_layer(inputs)
    return logits

def get_odin_score(inputs, model, args):
    # calculating the perturbation we need to add, i.e the sign of gradient of cross entropy loss w.r.t. input using simple FGSM attack perturbation
    epsilon = args.noise
    temperature = args.temp
    criterion = nn.CrossEntropyLoss()

    # get predicted labels
    inputs.requires_grad = True
    features = get_odin_feature(inputs, model, args)
    outputs = get_odin_logit(features, model, args)
    # outputs = model(inputs)
    labels = torch.argmax(outputs, dim = 1).to(inputs.device)

    # using temperature scaling
    outputs = outputs / temperature
    #back propagate loss
    loss = criterion(outputs, labels)
    loss.backward()

    # get pertubed image = image - noise*grad.sign
    gradient =  inputs.grad.data.sign()
    inputs = torch.add(inputs.data,  -epsilon*gradient)

    #get new softmax score
    model.eval()
    with torch.inference_mode():
        outputs = model(inputs)
    outputs = outputs / temperature
    scores, index = torch.max(torch.softmax(outputs, dim = 1), dim = 1)
    scores = scores.data.cpu().numpy()

    return scores