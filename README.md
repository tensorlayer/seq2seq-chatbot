# Seq2Seq Chatbot

This is a 200 lines implementation of Twitter/Cornell-Movie Chatbot, please read the following references before you read the code:

- [Practical-Seq2Seq](http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (optional)

### Prerequisites

- Python 
- [TensorFlow](https://github.com/tensorflow/tensorflow) >= 1.2
- [TensorLayer](https://github.com/zsdonghao/tensorlayer) >= 1.6.3

### Model

<table class="image">
<div align="center">
    <img src="http://suriyadeepan.github.io/img/seq2seq/seq2seq2.png"/>  
    <br>  
    <em align="center"></em>  
</div>
</table>

### Training

```
python main.py --batch-size 32 --num-epochs 50 -lr 0.001
```

### Inference

```
python main.py --inference-mode
```

### Results

```
Query > happy birthday have a nice day
 > thank you so much
 > thank babe
 > thank bro
 > thanks so much
 > thank babe i appreciate it
Query > donald trump won last nights presidential debate according to snap online polls
 > i dont know what the fuck is that
 > i think he was a racist
 > he is not a racist
 > he is a liar
 > trump needs to be president
