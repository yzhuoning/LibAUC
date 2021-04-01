

<p align="center">
  <img src="libauc.png" width="50%" align="center"/>
</p>

LibAUC
======
An end-to-end machine learning library for auc optimization.


Why is LibAUC?
---------------
Deep AUC Maximization (DAM) is a paradigm for learning a deep neural network by maximizing the AUC score of the model on a dataset. There are several benefits of maximizing AUC score over minimizing the standard losses, e.g., cross-entropy.

- In many domains, AUC score is the default metric for evaluating and comparing different methods. Directly maximizing AUC score can potentially lead to the largest improvement in the model’s performance.
- Many real-world datasets are usually imbalanced . AUC is more suitable for handling imbalanced data distribution since maximizing AUC aims to rank the predication score of any positive data higher than any negative data

Original Links
--------------

-  Repository: https://github.com/yzhuoning/libauc
-  Library website: https://libauc.org


How to install
--------------
```
$ pip install libauc
```

Example
-------
Plase run the following commands or check the demo code `train_cifar10_demo.ipynb`.

```shell
$ python
```
```python
>>> ...
>>> Losss = AUCMLoss(imratio=0.1)
>>> optimizer = PESG(model, a=Loss.a, b=Loss.b, alpha=Loss.alpha, imratio=0.1, lr=0.1, margin=0.9, gamma=500, weight_decay=1e-5)
>>> ...
>>> loss = Loss(y_pred, targets)
>>> optimizer.zero_grad()
>>> loss.backward(retain_graph=True)
>>> optimizer.step()
```

Citation
---------
If you use libauc in your work, please cite the following paper:
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
Please report issues/bugs to `yzhuoning@gmail.com`


Copyright
---------
Apache License 2.0

