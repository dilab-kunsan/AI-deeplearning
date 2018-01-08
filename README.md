# Deep Lerning Practice

Code for [book] (Authors)

## Requirements

- OS : Ubuntu 14.04 or higher

- GPU card(CUDA Compute Capability 3.0 or higher) (http://www.nvidia.co.kr/Download/index.aspx?lang=kr)

- CUDA toolkit 7.0 or higher (https://developer.nvidia.com/cuda-downloads)

- cuDNN v3 or higher (https://developer.nvidia.com/cudnn)

- Tensorflow 0.5.0 (https://www.tensorflow.org/)

## Dependencies

### Python 2.7.10

- numpy

- gensim

- fasttext 

- konlpy


## FFNN - News Article Category Classification using FFNN

- Data : 3,000 korean news article (Author crawled)

- Preprocess

``` python preprecess.py```

- Model

```python ffnn.py```


## MNIST - Handwriting Classiciation using CNN

- Data : MNIST dataset (http://yann.lecun.com/exdb/mnist/)

- Model

```python mnist_conv.py```
	

## RNN - Translation using Seq2Seq Model

- Reference : Tensorflow Tutorial (https://www.tensorflow.org/tutorials/seq2seq)

- Data : 10^9French-English corpus and Development sets (http://www.statmt.org/wmt10/)

- Model : Seq2Seq Model (https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate)

