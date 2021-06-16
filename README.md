<p align="center">
  <img src="https://github.com/yzhuoning/LibAUC/blob/main/imgs/libauc.png" width="50%" align="center"/>
</p>


LibAUC
======
An end-to-end machine learning library for AUC optimization (<strong>AUROC, AUPRC</strong>). 

Why LibAUC?
---------------
Deep AUC Maximization (DAM) is a paradigm for learning a deep neural network by maximizing the AUC score of the model on a dataset. There are several benefits of maximizing AUC score over minimizing the standard losses, e.g., cross-entropy.

- In many domains, AUC score is the default metric for evaluating and comparing different methods. Directly maximizing AUC score can potentially lead to the largest improvement in the model’s performance.
- Many real-world datasets are usually imbalanced. AUC is more suitable for handling imbalanced data distribution since maximizing AUC aims to rank the predication score of any positive data higher than any negative data

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
- 02.Optimizing <strong>AUROC</strong> loss with ResNet20 on Imbalanced CIFAR10 [[Notebook](https://github.com/yzhuoning/LibAUC/blob/main/examples/02_Optimizing_AUROC_with_ResNet20_on_Imbalanced_CIFAR10.ipynb)][[Script](https://github.com/yzhuoning/LibAUC/tree/main/examples/scripts)]
- 03.Optimizing <strong>AUPRC</strong> loss with ResNet18 on Imbalanced CIFAR10 [[Notebook](https://github.com/yzhuoning/LibAUC/blob/main/examples/03_Optimizing_AUPRC_with_ResNet18_on_Imbalanced_CIFAR10.ipynb)][[Script](https://github.com/yzhuoning/LibAUC/tree/main/examples/scripts)]
- 04.Training with Pytorch Learning Rate Scheduling [[Notebook](https://github.com/yzhuoning/LibAUC/blob/main/examples/04_Training_with_Pytorch_Learning_Rate_Scheduling.ipynb)][[Script](https://github.com/yzhuoning/LibAUC/tree/main/examples/scripts)]
- 05.Training with Imbalanced Datasets on Distributed Setting [[Coming soon]()]

### Quickstart for beginner:
#### Optimizing AUROC (Area Under the Receiver Operating Characteristic)
```python
>>> #import library
>>> from libauc.losses import AUCMLoss
>>> from libauc.optimizers import PESG
...
>>> #define loss
>>> Loss = AUCMLoss()
>>> optimizer = PESG()
...
>>> #training
>>> model.train()    
>>> for data, targets in trainloader:
>>>	data, targets  = data.cuda(), targets.cuda()
        preds = model(data)
        loss = Loss(preds, targets) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
...	
>>> #restart stage
>>> optimizer.update_regularizer()		


```
#### Optimizing AUPRC (Area Under the Precision-Recall Curve)
```python
>>> #import library
>>> from libauc.losses import APLoss_SH
>>> from libauc.optimizers import SOAP_SGD, SOAP_ADAM
...
>>> #define loss
>>> Loss = APLoss_SH()
>>> optimizer = SOAP_SGD()
...
>>> #training
>>> model.train()    
>>> for index, data, targets in trainloader:
>>>	data, targets  = data.cuda(), targets.cuda()
        preds = model(data)
        loss = Loss(preds, targets, index) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()	

```



Please visit our [website](https://libauc.org/) or [github](https://github.com/yzhuoning/libAUC) for more examples. 

Citation
---------
If you find LibAUC useful in your work, please cite the following paper for our library:
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
If you have any questions, please contact us @ [Zhuoning Yuan](https://homepage.divms.uiowa.edu/~zhuoning/) [yzhuoning@gmail.com] and [Tianbao Yang](https://homepage.cs.uiowa.edu/~tyng/) [tianbao-yang@uiowa.edu] or please open a new issue in the Github. 
