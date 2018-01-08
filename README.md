# Deep Lerning Practice

Code for [book] (BW On, JJ Kim, IH Song, SY Oh)

## Requirements

- OS : Ubuntu 14.04 or higher

- GPU card(CUDA Compute Capability 3.0 or higher)(http://www.nvidia.co.kr/Download/index.aspx?lang=kr)

- CUDA toolkit 7.0 or higher(https://developer.nvidia.com/cuda-downloads)

- cuDNN v3 or higher(https://developer.nvidia.com/cudnn)

- Tensorflow(https://www.tensorflow.org/)


## FFNN - News Article Category Classification using FFNN

- Data : 3,000 korean news article(Author crawled)

- Preprocess

``` python preprecess.py```

- Model

```python ffnn.py```


## MNIST - Handwriting Classiciation using CNN

- Data : MNIST dataset(http://yann.lecun.com/exdb/mnist/)

- Model

```python mnist_conv.py```
	

## RNN - Translation using Seq2Seq Model(*Reference : https://www.tensorflow.org/tutorials/seq2seq)

- Data : 10^9French-English corpus and Development sets (http://www.statmt.org/wmt10/)

- Model : Seq2Seq Model (https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate)
