from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import pdb
from tqdm import tqdm

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc


def siglip_classifier(classnames, template, siglip_model, processor):
    with torch.no_grad():
        siglip_weights = []

        for classname in tqdm(classnames):
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            # prompt ensemble for ImageNet
            
            inputs = processor(text=texts, padding="max_length", return_tensors="pt")
            class_embeddings = siglip_model.get_text_features(**inputs)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            siglip_weights.append(class_embedding)

        siglip_weights = torch.stack(siglip_weights, dim=1).cuda()
    return siglip_weights


def build_cache_model(cfg, clip_model, train_loader_cache):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, siglip_model, processor, loader):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                inputs = processor(images=images, return_tensors="pt")
                with torch.no_grad():
                    image_features = siglip_model.get_image_features(**inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)
        torch.save(features, os.path.join(cfg['cache_dir'], cfg['model'] + "_f.pt"))
        torch.save(labels, os.path.join(cfg['cache_dir'], cfg['model'] + "_l.pt"))
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def siglip_zero_shot(cfg, siglip_model, processor, imagenet, test_loader):
    # Textual features
    print("Getting textual features as Siglip's classifier.")
    siglip_weights = siglip_classifier(imagenet.classnames, imagenet.template, siglip_model, processor)

    # Pre-load test features
    print("\nLoading visual features and labels from test set.")
    test_features, test_labels = pre_load_features(cfg, "test", siglip_model, processor, test_loader)
    siglip_logits = 100. * test_features.cuda() @ siglip_weights.cuda()
    acc = cls_acc(siglip_logits, test_labels.cuda())
    print("\n**** Zero-shot Siglip's test accuracy: {:.2f}. ****\n".format(acc))