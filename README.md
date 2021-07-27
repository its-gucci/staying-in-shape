## Staying in Shape: Learning Invariant Shape Representations using Contrastive Learning

This is the original PyTorch implementation of the [Staying in Shape paper](https://arxiv.org/abs/2107.03552):
```
@article{gu2021staying,
  title={Staying in Shape: Learning Invariant Shape Representations using Contrastive Learning},
  author={Gu, Jeffrey and Yeung, Serena},
  journal={arXiv preprint arXiv:2107.03552},
  year={2021}
}
```

### Preparation

Download the aligned version of ModelNet40 [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and ShapeNet from [here](https://shapenet.org/). You may also need to add ```models``` folder to the base directory. 

### Unsupervised Training

An example training command for the unsupervised pre-training of our model is 
```
python main_moco_shape.py \ 
  [your shapenet folder] -d ShapeNet \
  --lr 0.0075 \
  --batch-size 64 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --model-name [your model name] --orth 
```
The available augmentations are ```--orth```,```--rip```, ```--perturb```, ```--interp```, ```--rotation```, ```--y-rotation```, which are described in the paper. Multiple data augmentation settings in the paper uses the ```--rand``` flag, which applies a random augmentation out of the augmentations provided to the model, as opposed to sequentially. Models are saved in ```models/```. 


### ModelNet40 Classification

With a pre-trained model, to train a supervised MLP classifier, run:
```
python main_lincls.py \
  [your path to modelnet40] \
  --lr 0.01 \
  --batch-size 128 \
  --pretrained [your checkpoint path]/checkpoint_0199.pth.tar \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp \
  --model-name [your classification model name]
```
To run robustness experiments, the same augmentation flags are available as in the Unsupervised Training section. 

### See Also
This repository is based on [this implementation](https://github.com/facebookresearch/moco) of the [MoCo paper](https://arxiv.org/abs/1911.05722) and [MoCo v2 paper](https://arxiv.org/abs/2003.04297):
```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```
We also based some code off [this implementation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch) of PointNets.
```
@article{Pytorch_Pointnet_Pointnet2,
      Author = {Xu Yan},
      Title = {Pointnet/Pointnet++ Pytorch},
      Journal = {https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
      Year = {2019}
}
```
