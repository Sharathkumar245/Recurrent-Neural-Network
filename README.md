# assignment3_dl
In this assignment we basically try to predict the translated word from one language to another language. So baically this is called as transliteration in more technoncal terms. Here we used a dataset set **(aksharantar)** 


# Description about the model that we have designed.
Here for this assignment convertion from the language that we selected is the from "English" to "telugu" from the dataset **akshantar**.
To reach the final goal that is to predict the data or transliteration of the given input language to other language, we will design the Recurrent Neural Network(**RNN**) Model and also implemented the varients of the this Recurrent Neural Network like the Long - Short Term memory and Gated Recurrent networks.

# Attention and without Attention
We have added here the two different kinds of classes for the Decoder that which will consider the Attention in one and other without attention and the classes are like **attention_add_decoder** attention added decoder and **Decoder** for without attention.

# Creation of the vocabulary for each language (inputs and outputs of the points)
In this the data that was provided in the data is the form of langauge and to convert it into vectors, we should first make the vocabulary for this by traversing the whole dataset for each train dataset and find the vocabulary elements.


# Convertion of the given point to the vectors
In this we need to traverse to the every point in the given dataset and the convert those data points to the vectors that and the values with which we are creating these vectors are according to the corresponding elements and the positions of that particular letter in the vocabulary list of their respective languages.
To make the vector with feasible sizes we need to find the maximum data point (data) size.
By finding the maximum length to make the vectors not to be dimensionally mismatching when the operations are applying on it we added the remaining bits 
