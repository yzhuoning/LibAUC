.. -*- mode: rst -*-

LibAUC
======

Description
-----------

AN END-TO-END MACHINE LEARNING LIBRARY FOR AUC OPTIMIZATION.

Why is LibAUC?
---------------
Deep AUC Maximization (DAM) is a paradigm for learning a deep neural network by maximizing the AUC score of the model on a dataset. There are several benefits of maximizing AUC score over minimizing the standard losses, e.g., cross-entropy.

- In many domains (e.g., medical diagonosis) the AUC score is the default metric for evaluating and comparing different methods. Directly maximizing AUC score can potentially lead to the largest improvement in the model’s performance.Many real-world datasets are usually imbalanced (e.g., the number of malignant cases is usually much less than benign cases). AUC is more suitable for handling imbalanced data distribution since maximizing AUC aims to rank the predication score of any positive data higher than any negative data
- Many real-world datasets are usually imbalanced (e.g., the number of malignant cases is usually much less than benign cases). AUC is more suitable for handling imbalanced data distribution since maximizing AUC aims to rank the predication score of any positive data higher than any negative data
- 
Original Links
--------------

-  Repository: https://github.com/yzhuoning/libauc
-  Library website: https://libauc.org

Purpose of this package
-----------------------

The idea behind this package is to use the same code as in
https://github.com/yzhuoning/libauc using the very convenient pip command

How to install
--------------

::

   pip install libauc

Example
-------

1. Download https://github.com/cjlin1/libsvm/blob/master/heart_scale
   file.
2. Run the following commands

::

   >>> from libsvm.svmutil import *
   >>> y, x = svm_read_problem('path/to/heart_scale')
   >>> m = svm_train(y[:200], x[:200], '-c 4')
   *.*
   optimization finished, #iter = 257
   nu = 0.351161
   obj = -225.628984, rho = 0.636110
   nSV = 91, nBSV = 49
   Total nSV = 91
   >>> p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)
   Accuracy = 84.2857% (59/70) (classification)


Copyright
---------

Copyright (c) 2000-2018 Chih-Chung Chang and Chih-Jen Lin All rights
reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither name of copyright holders nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Maintainer
----------

-  Zhuoning Yuan  `yzhuoning@gmail.com`_

.. yzhuoning@gmail.com: yzhuoning@gmail.com
