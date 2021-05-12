<p align="center">
  <img src="https://github.com/yzhuoning/LibAUC/blob/main/imgs/libauc.png" width="50%" align="center"/>
</p>


LibAUC
======
An end-to-end machine learning library for AUC optimization. 

Why LibAUC?
---------------
Deep AUC Maximization (DAM) is a paradigm for learning a deep neural network by maximizing the AUC score of the model on a dataset. There are several benefits of maximizing AUC score over minimizing the standard losses, e.g., cross-entropy.

- In many domains, AUC score is the default metric for evaluating and comparing different methods. Directly maximizing AUC score can potentially lead to the largest improvement in the model’s performance.
- Many real-world datasets are usually imbalanced . AUC is more suitable for handling imbalanced data distribution since maximizing AUC aims to rank the predication score of any positive data higher than any negative data

Links
--------------
-  Repository: https://github.com/yzhuoning/libauc
-  Website: https://libauc.org


Installation
--------------
```
$ pip install libauc
```

Usage
-------
### Official Tutorials:
- 01.Creating Imbalanced Benchmark Datasets [[Notebook](https://github.com/yzhuoning/LibAUC/blob/main/examples/01_Creating_Imbalanced_Benchmark_Datasets.ipynb)][[Script](https://github.com/yzhuoning/LibAUC/tree/main/examples/scripts)]
- 02.Training ResNet20 with Imbalanced CIFAR10 [[Notebook](https://github.com/yzhuoning/LibAUC/blob/main/examples/02_Training_ResNet20_with_Imbalanced_CIFAR10.ipynb)][[Script](https://github.com/yzhuoning/LibAUC/tree/main/examples/scripts)]
- 03.Training with Pytorch Learning Rate Scheduling [[Notebook](https://github.com/yzhuoning/LibAUC/blob/main/examples/03_Training_with_Pytorch_Learning_Rate_Scheduling.ipynb)][[Script](https://github.com/yzhuoning/LibAUC/tree/main/examples/scripts)]
- 04.Training with Imbalanced Datasets on Distributed Setting [[Coming soon]()]

### Quickstart for beginner:
```python
>>> #import library
>>> from libauc.losses import AUCMLoss
>>> from libauc.optimizers import PESG
...
>>> #define loss
>>> Loss = AUCMLoss(imratio=0.1)
>>> optimizer = PESG(imratio=0.1)
...
>>> #training
>>> model.train()    
>>> for data, targets in trainloader:
>>>	data, targets  = data.cuda(), targets.cuda()
        preds = model(data)
        loss = Loss(preds, targets) 
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
...	
>>> #restart stage
>>> optimizer.update_regularizer()		
...   
>>> #evaluation
>>> model.eval()    
>>> for data, targets in testloader:
	data, targets  = data.cuda(), targets.cuda()
        preds = model(data)

```

Please visit our [website](https://libauc.org/) or [github](https://github.com/yzhuoning/libAUC) for more examples. 

Citation
---------
If you find LibAUC useful in your work, please cite the following paper:
```
@article{yuan2020robust,
title={Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification},
author={Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao},
journal={arXiv preprint arXiv:2012.03173},
year={2020}
}
```

Contact
----------
If you have any questions, please contact us @ Zhuoning Yuan [yzhuoning@gmail.com] and Tianbao Yang [tianbao-yang@uiowa.edu] or please open a new issue in the Github. 
