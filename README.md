# Question Answering System with BERT



### Introduction

In this notebook we are trying to build a Question Answering system with BERT on the Squad Dataset 

#### BERT

[BERT ](https://arxiv.org/abs/1810.04805) was released on 11th Oct 2019 by Google. BERT is a Bidirectional Transformer (basically an encode-only ) with a [Masked Language Modelling](https://www.machinecurve.com/index.php/question/what-is-a-masked-language-model-mlm-objective/) and [Next Sentence Prediction](https://www.machinecurve.com/index.php/question/what-is-the-next-sentence-prediction-nsp-language-objective/) task, where the goal is to predict the missing samples. So Given A_C_E, predict B and D.

BERT makes use of Transformer architecture (attention mechanism) that learns contextual relations between words in a text. BERT falls into a self-supervised model category. That means, it can generate inputs and outputs from the raw corpus without being explicitly programmed by humans. Since BERT's goal is to generate a language model, only the encoder mechanism is necessary.

As opposed to directional models, which read the text input sequentially (left to right or right to left), the Transformer encoder reads the entire sequence of words at once. Therefore it is considered bidirectional. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word).

![p1_bert_highlevel](F:\Git\Question-Answering-with-BERT\README.assets\p1_bert_highlevel.PNG)

The diagram above is a high-level Transformer encoder. The input is a sequence of tokens, which are first embedded into vectors and then processed in the neural network. The output is a sequence of vectors, in which each vector corresponds to an input token with the same index.

When training language models, there is a challenge of defining a prediction goal (self-supervision).To overcome this challenge, BERT uses two training strategies.

##### **MASKED LANGUAGE MODEL**

Before feeding word sequence into BERT, 15% of the words in each sequence are replaced with a [MASK] token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked words in the sequence. In technical terms, the prediction of the output words requires:

1. Adding a classification layer on top of the encoder output
2. Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
3. Calculating the probability of each word in the vocabulary with softmax. 

**NEXT SENTENCE PREDICTION**

In the BERT training process, the model receives pairs of sentences as input and learns to predict if the second sentence in the pair *is* the subsequence sentence in the original document. During training, 50% of the inputs are a pair in which the second sentence is the subsequence sentence in the original document, while in the other 50% a random sentence from the corpus is chosen. The assumption is that the random sentence will be disconnected from the first sentence. 

To help the model distinguish between the two sentences in training, the input is processed in the following way before entering the model:

1. A [CLS] token is inserted at the beginning of the first sentence and a [SEP] token is inserted at the end of each sentence. 
2. A sentence embedding indicating Sentence A or Sentence B is added to each token. Sentence embeddings are similar in concept to token embedding with a vocabulary of 2. 
3. A positional embedding is added to each token to indicate its position in the sequence.

 To predict the second sentence is indeed connected to the first, the following steps are performed:

1. The entire input sequence goes through the transformer
2. The output of the [CLS] token is transformed into a 2x1 shaped vector, using a simple classification layer (learned matrices of weights and biases).
3. Calculating the probability of IsNextSequence with SoftMax

![p1_bert](F:\Git\Question-Answering-with-BERT\README.assets\p1_bert.PNG)

While training the BERT model, Masked LM and NSP are trained together, with the goal of maximizing the combined loss function of the two strategies. 

The BERT loss function takes into consideration only the prediction of the masked values and ignores the prediction of the non-masked words (this makes solving this problem even harder as we have reduced the supervision further). As a consequence, the model converges slower than directional models, a characteristic that is offset by its increased context-awareness. 

#### Training Logs

![p1_training_logs](F:\Git\Question-Answering-with-BERT\README.assets\p1_training_logs.PNG)

#### Training loss

<img src="F:\Git\Question-Answering-with-BERT\README.assets\p1_training_loss.PNG" alt="p1_training_loss" style="zoom:75%;" />
**Sample Results**

```
question       >> When was Dali defended by the Yuan?
predicted answer >> 1253

question       >> What molecules of the adaptive immune system only exist in jawed vertebrates?
predicted answer >> immunoglobulins and T cell receptors

question       >> What does the capabilities approach look at poverty as a form of?
predicted answer >> capability deprivation

question       >> How much can the SP alter income tax in Scotland?
predicted answer >> up to 3 pence in the pound

question       >> The French thought bringing what would uplift other regions?
predicted answer >> Christianity and French culture

question       >> What organization is the IPCC a part of?
predicted answer >> World Meteorological Organization

question       >> At what pressure is water heated in the Rankine cycle?
predicted answer >> high pressure

question       >> What limits the Rankine cycle's efficiency?
predicted answer >> the working fluid

question       >> In what year did Joseph Priestley recognize oxygen?
predicted answer >> 1774
```

refer to complete solution 👉 [here](https://github.com/krishnarevi/TSAI_END2.0_Session14/blob/main/part1_session14.ipynb).



