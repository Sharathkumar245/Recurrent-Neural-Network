# DL_assignment_3
In this assignment we basically try to predict the translated word from one language to another language. So baically this is called as transliteration in more technoncal terms. Here we used a dataset set **(aksharantar)** 
# How to run the codes
Here we are providing two ways like .ipynb and .py files these two can be executed one online and like the inputs in the .ipynb files are providing using the **Wandb.ai** form this we can configure the input parameters to the code, wandb can provide us the plots for the accuracies and losses and this will help to get the idea about how the model is behaving according to the different parameters, also provide the correlations betweeen the parameters and the goals of the model that we are training for.<br>
The second file .py can be able to run on the local device by giving the particular locations(paths) to the dataset and we can execute the code in our local system this can also be run in the colab or any other .py file supportive compilers too.
# libraries we used
Here in this the **PyTorch** library will allow us to make the operations much more easier by letting us using the inbuilt functions and we don't need to write the codes from scratch. These operations also took as minimum time by providing optimized snippets for those inbuilt functions. <br>
We used **Wandb.ai** for the plots that which will help us to understand how the parameters in the model are correlated and provide us a clear visual representations like graphs and plots. For this we need to connect to the wandb with a private key(activation key). <br>
# Description about the model that we have designed.
Here for this assignment convertion from the language that we selected is the from "English" to "telugu" from the dataset **akshantar**.<br>
To reach the final goal that is to predict the data or transliteration of the given input language to other language, we will design the Recurrent Neural Network(**RNN**) Model and also implemented the varients of the this Recurrent Neural Network like the Long - Short Term memory and Gated Recurrent networks.<br>

# Attention and without Attention
We have added here the two different kinds of classes for the Decoder that which will consider the Attention in one and other without attention and the classes are like **attention_add_decoder** attention added decoder and **Decoder** for without attention.<br>

# Creation of the vocabulary for each language (inputs and outputs of the points)
In this the data that was provided in the data is the form of langauge and to convert it into vectors, we should first make the vocabulary for this by traversing the whole dataset for each train dataset and find the vocabulary elements.<br>


# Convertion of the given point to the vectors
In this we need to traverse to the every point in the given dataset and the convert those data points to the vectors that and the values with which we are creating these vectors are according to the corresponding elements and the positions of that particular letter in the vocabulary list of their respective languages.<br>
To make the vector with feasible sizes we need to find the maximum data point (data) size.<br>
By finding the maximum length to make the vectors not to be dimensionally mismatching when the operations are applying on it we added the remaining bits dummy with zeros as the values at the respective free slots this makes the vector to be same size for all the data points as we take the maximum sized input to the create the vectors. <br>

# Encoder 
We decleared encoder as a class and it will take all the arguements that are necessary to like then number of encoder layers, number of decoder layers, number of batch size, embedding size, cell type (this tells that what kind of varient of RNN we want to use like GRU, RNN, LSTM), bi-directional, these are some of the parameters that we need to consider of the encoder.<br>
This will take the input data (english language word(vectored representation)) and do the forward propagation according to the cell type that we selected this tells the information. <br>
After the forward propagation is done we will take the we return the outputs and the hidden states (h) according to the cell types. <br>
These can be used as the inputs to the decoder and then the decoder will produce the outputs ( targeted word representation of the given input word). <br>

# Decoder
We decleared this decoder as a class and in the same way as the encoder, decoder too takes the inputs and then produce the inputs like the decoder layers and all the things. <br>
To the decoder the output of the encoder is also considered as the one of the inputs, In the case of **Without attention** this will not consider the relative relations between the letter and this will make the model to be somewhat less efficient than the model in which we are considering the **attention** feature in the code.<br>

# attention added decoder
From the above discussion we came to get some kind of understanding that if we know the relative relation then it would be much more effecient and better model.<br>
for the same we created another class like the attention_added_decoder.<br>

# Sequence to Sequence 
This will help to merge these two RNN modelled components (Encoders, Decoders). <br>
This enables the model to enable the model to train in the pair of the input(sequence of letters(vectored representation)) this will let the other parameters of both encoders and decoders learn, while minimizing the loss. <br>
This sequence to sequence are used cause we are dealing with the different lengthed sequences. <br>
This return the model that was trained on the particular train data that we provided. <br>

# Created the CSV files for the predictions
In this after the model is trained we can be able to use that trained model on the validation dataset or test data set.<br>
To create the CSV files we imported the library for CSV. <br>
Now when we trained our model and the output values are also vectors of number according to the indeces of the vocabulary of the particular langauge.<br>
We then called another function **vector_to_actual_words** this function will help to convert the vectorial representation of the output back to the actual languagial representation and this function return the list of the input and actual and the predicted words.<br> 
After converting back to the letters this will be written into the CSV files that were we have given the path. <br>

