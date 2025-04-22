# VFM_lite_evaluation
This is repo for VFM-lite evaluation


# install for open_clip:
```shell
pip install git+https://github.com/openai/CLIP.git
pip install -r requirement.txt
```

# run for open_clip zero shot evaluation:
```shell
python eval_imagenet.py --config configs/imagenet.yaml
```

The result is:
**** Zero-shot CLIP's test accuracy: 63.75. ****


This code is based on [Tip-adapter](https://github.com/gaopengcuhk/Tip-Adapter)