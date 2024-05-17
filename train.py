# !pip install torchvision
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import pandas as pd
import random
import argparse

from tqdm import tqdm
#device selection CPU or GPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

train_data_root='/content/drive/MyDrive/Deep_learning_a3/aksharantar_sampled/tel/tel_train.csv'
valid_data_root='/content/drive/MyDrive/Deep_learning_a3/aksharantar_sampled/tel/tel_valid.csv'
test_data_root='/content/drive/MyDrive/Deep_learning_a3/aksharantar_sampled/tel/tel_test.csv'

train_data=pd.read_csv(train_data_root,header=None)
valid_data=pd.read_csv(valid_data_root,header=None)
test_data=pd.read_csv(test_data_root,header=None)
# print(train_data[0][1])
eng_list_train=train_data[0]
tel_list_train=train_data[1]
eng_list_valid=valid_data[0]
tel_list_valid=valid_data[1]
eng_list_test=test_data[0]
tel_list_test=test_data[1]
# print(tel_list_train)

eng_vocab=[] #list of the letters in the english words
tel_vocab=[] #list of the letters in the telugu words
max_eng_len=-1
max_tel_len=-1
max_eng_word=""
max_tel_word=""
for word in eng_list_train:
    max_eng_len=max(max_eng_len,len(word))
    if(max_eng_len==len(word)):
        max_eng_word=word
    for letter in word:
        eng_vocab.append(letter)
eng_vocab=list(set(eng_vocab))
eng_vocab.sort()

for word in tel_list_train:
    max_tel_len=max(max_tel_len,len(word))
    if(max_tel_len==len(word)):
        max_tel_word=word
    for letter in word:
        tel_vocab.append(letter)
tel_vocab=list(set(tel_vocab))
tel_vocab.sort()

# this is done to get the maximum that a data point size we did not store any thing in the vocabulary from these traversals
for word in eng_list_valid:
    max_eng_len=max(max_eng_len,len(word))
for word in eng_list_test:
    max_eng_len=max(max_eng_len,len(word))
for word in tel_list_test:
    max_tel_len=max(max_tel_len,len(word))
for word in tel_list_valid:
    max_tel_len=max(max_tel_len,len(word))
# print(tel_vocab)
# print(eng_vocab)
# print(len(max_tel_word))
# print(len(max_eng_word))
# print(max_tel_len,max_eng_len)
# print(eng_vocab)
# print(tel_vocab)
# print(len(tel_vocab))
# print(len(eng_vocab))


# function to convert the telugu or english word to a vectorial representation
def word_to_vector(language,word):
    vec=[]
    if(language=="english"):
        vec.append(len(eng_vocab)+1)
        for letter in word:
            for albt in range(len(eng_vocab)):
                if(eng_vocab[albt]==letter):
                    vec.append(albt+1)
        while(len(vec)<(max_eng_len+1)):
            vec.append(0)
        vec.append(0)
    else:
        vec.append(len(tel_vocab)+1)
        for letter in word:
            for albt in range(len(tel_vocab)):
                if(tel_vocab[albt]==letter):
                    vec.append(albt+1)
        while(len(vec)<(max_tel_len+1)):
            vec.append(0)
        vec.append(0)
    return vec
# print(word_to_vector("english",eng_list_train[1]))
# print(word_to_vector("telugu",tel_list_train[1]))
# print(len(word_to_vector("english",eng_list_train[1])))
# print(len(word_to_vector("telugu",tel_list_train[1])))
# print(eng_list_train[1])
# print(tel_list_train[1])
# print(eng_list_train.shape)
# print(len(eng_vocab))
# print(len(tel_vocab))




#creation of the matrix for the english and the telugu words

# for training data
eng_matrix_train=[]
for word in eng_list_train:
    eng_matrix_train.append(word_to_vector("english",word))
tel_matrix_train=[]
for word in tel_list_train:
    tel_matrix_train.append(word_to_vector("telugu",word))
eng_matrix_train=torch.tensor(eng_matrix_train)
tel_matrix_train=torch.tensor(tel_matrix_train)
# print(eng_matrix_train[1])
# print(tel_matrix_train[1])
# print(eng_list_train[1])
# print(tel_list_train[1])


#for validation data
eng_matrix_valid=[]
tel_matrix_valid=[]
for word in eng_list_valid:
    eng_matrix_valid.append(word_to_vector("english",word))
for word in tel_list_valid:
    tel_matrix_valid.append(word_to_vector("telugu",word))
eng_matrix_valid=torch.tensor(eng_matrix_valid)
tel_matrix_valid=torch.tensor(tel_matrix_valid)
# print(eng_matrix_valid[1])
# print(tel_matrix_valid[1])
# print(eng_list_valid[1])
# print(tel_list_valid[1])


# for test data
eng_matrix_test=[]
tel_matrix_test=[]
for word in eng_list_test:
    eng_matrix_test.append(word_to_vector("english",word))
for word in tel_list_test:
    tel_matrix_test.append(word_to_vector("telugu",word))
eng_matrix_test=torch.tensor(eng_matrix_test) # converted to tensors for the tensor feasible applications
tel_matrix_test=torch.tensor(tel_matrix_test)
# print(eng_matrix_test[1])
# print(tel_matrix_test[1])
# print(eng_list_test[1])
# print(tel_list_test[1])


# print(len(eng_matrix_train))
# print(len(eng_matrix_valid))
# print(len(eng_matrix_test))
# print(tel_matrix_train.shape)
# print(eng_matrix_train.shape)




# encoder class for the input and producing the output as the input for the decoder
class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,enc_layers,hidden_size,cell_type,bi_directional_bit,dropout,batch_size):
        super(Encoder,self).__init__()
        self.input_size=input_size
        self.embedding_size=embedding_size
        self.enc_layers=enc_layers
        self.cell_type=cell_type
        self.bi_directional_bit=bi_directional_bit
        self.dropout=nn.Dropout(dropout)
        self.embedding=nn.Embedding(input_size,embedding_size)
        self.hidden_size=hidden_size
        self.batch_size=batch_size
        #based on the type of the cell considering the variant of the neural network to be used
        if(cell_type=="RNN"):    
            self.rnn=nn.RNN(embedding_size,hidden_size,enc_layers,dropout=dropout,bidirectional=bi_directional_bit)
        elif(cell_type=="GRU"):
            self.gru=nn.GRU(embedding_size,hidden_size,enc_layers,dropout=dropout,bidirectional=bi_directional_bit)
        else:
            self.lstm=nn.LSTM(embedding_size,hidden_size,enc_layers,dropout=dropout,bidirectional=bi_directional_bit)
            
    
            
    #forward passing 
    def forward(self,x,hidden,cell):
        embedding=self.embedding(x).view(-1,self.batch_size,self.embedding_size)
#         embedding=self.dropout(embedding) # adding the dropout at the input layer
        if(self.cell_type=="RNN"):
            output,hidden=self.rnn(embedding,hidden)
        elif(self.cell_type=="GRU"):
            output,hidden=self.gru(embedding,hidden)
        else:
            output,(hidden,cell)=self.lstm(embedding,(hidden,cell))
            return output,hidden,cell
        return output,hidden
    
    # initialize the tensor to zeroes at start
    def initialize_hidden(self):
        if(self.bi_directional_bit==True):
            return torch.zeros(2*self.enc_layers,self.batch_size,self.hidden_size,device=device)
        return torch.zeros(self.enc_layers,self.batch_size,self.hidden_size,device=device)
    def initialize_cell(self):
        if(self.bi_directional_bit==True):
            return torch.zeros(2*self.enc_layers,self.batch_size,self.hidden_size,device=device)
        return torch.zeros(self.enc_layers,self.batch_size,self.hidden_size,device=device)
    

#declearing the decoder class
class Decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,dec_layers,dropout,cell_type,output_size):
        super(Decoder,self).__init__()
        self.input_size=input_size
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.dec_layers=dec_layers
        self.dropout=nn.Dropout(dropout)
        self.cell_type=cell_type
        self.embedding=nn.Embedding(input_size,embedding_size)
        if(cell_type=="RNN"):
            self.rnn=nn.RNN(embedding_size,hidden_size,dec_layers,dropout=dropout)
        elif(cell_type=="GRU"):
            self.gru=nn.GRU(embedding_size,hidden_size,dec_layers,dropout=dropout)
        else:
            self.lstm=nn.LSTM(embedding_size,hidden_size,dec_layers,dropout=dropout)
        self.fully_conc=nn.Linear(hidden_size,output_size)
        
    # forward pass
    def forward(self,x,prev_output,prev_hidden,cell=0):
        x=x.unsqueeze(0).int() # convert the input token X to tensor and gives a single dimension
        embedding=self.embedding(x)
        embedding=self.dropout(embedding)
        if(self.cell_type=="RNN"):
            outputs,hidden=self.rnn(embedding,prev_hidden)
        elif(self.cell_type=="GRU"):
            outputs,hidden=self.gru(embedding,prev_hidden)
        else:
            outputs,(hidden,cell)=self.lstm(embedding,(prev_hidden,cell))
        
        #as we converted it by using unsqueezing the dimension of the output is like(1,N,hidden_size)
        pred=self.fully_conc(outputs)
        # this makes the dimension as the (1,N,size_of_vocab)
        pred=pred.squeeze(0)
        # this makes the dimension as the (N,size_of_vocab)
        if(self.cell_type=="GRU" or self.cell_type == "RNN"):
            return pred,hidden
        return pred,hidden,cell
    
    
    #initialize the tensor to zeroes
    def initialize_hidden(self):
        return torch.zeros(self.dec_layers,self.batch_size,self.hidden_size,device=device)
    

class attention_add_decoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,output_size,cell_type,dec_layers,dropout,bi_directional_bit):
        super(attention_add_decoder,self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.cell_type=cell_type
        self.dec_layers=dec_layers
        self.bi_directional_bit=bi_directional_bit
        self.cell_type=cell_type
        self.embedding_size=embedding_size
        self.dropout=nn.Dropout(dropout)
        self.length=len(eng_matrix_train[0])
        self.embedding=nn.Embedding(input_size,embedding_size)
        if(self.cell_type=="LSTM"):
            self.lstm=nn.LSTM(hidden_size,hidden_size,dec_layers,dropout=dropout)
        elif(self.cell_type=="GRU"):
            self.gru=nn.GRU(hidden_size,hidden_size,dec_layers,dropout=dropout)
        else:
            self.rnn=nn.RNN(hidden_size,hidden_size,dec_layers,dropout=dropout)
        self.fully_conc=nn.Linear(hidden_size,output_size)
        self.attention=nn.Linear(hidden_size+embedding_size,self.length)
        if(bi_directional_bit==True):
            self.atten_adding=nn.Linear(hidden_size*2+embedding_size,hidden_size)
        elif(bi_directional_bit==False):
            self.atten_adding=nn.Linear(hidden_size+embedding_size,hidden_size)
        
    def forward(self,x,prev_output,prev_hidden,cell=0):
        prev_output = prev_output.permute(1,0,2) #this will help to rearrange the dimensions to 1,0,2
        x=x.unsqueeze(0)
        embedded=self.dropout(self.embedding(x))
        cnct=torch.cat((embedded[0],prev_hidden[0]),1)
        atten_parameters=func.softmax(self.attention(cnct),dim=1)
        atten_added=torch.bmm(atten_parameters.unsqueeze(1),prev_output) # this performs a batch wise matrix multiplications4
        atten_added=atten_added.squeeze(1)
#         print("Hello")
#         print("shape of embedded[0] is: ",embedded[0].shape)
#         print("shape of atten_added is: ",atten_added.shape)
        final_output=torch.cat((embedded[0],atten_added),1)
        final_output=torch.unsqueeze(self.atten_adding(final_output),0)
        final_output=func.relu(final_output)
        
        if(self.cell_type=="RNN"):
            outputs,hidden=self.rnn(final_output,prev_hidden)
        elif(self.cell_type=="GRU"):
            outputs,hidden=self.gru(final_output,prev_hidden)
        else:
            outputs,(hidden,cell)=self.lstm(final_output,(prev_hidden,cell))
        
        pred= self.fully_conc(outputs)
        pred= pred.squeeze(0)
        if(self.cell_type=="GRU" or self.cell_type=="RNN"):
            return pred,hidden
        else:
            return pred,hidden,cell
        


class Seq_to_seq(nn.Module):
    def __init__(self,decoder,encoder,cell_type,bidirectional_bit,encoder_layers,decoder_layers):
        super(Seq_to_seq,self).__init__()
        self.decoder=decoder
        self.encoder=encoder
        self.cell_type=cell_type
        self.bidirectional_bit=bidirectional_bit
        self.encoder_layers=encoder_layers
        self.decoder_layers=decoder_layers
        
    #forward pass
    def forward(self,input_seq,target,teacher_force_ratio=0.5):
        batch_size=input_seq.shape[1]
        tar_seq_length=target.shape[0]
        final_target_vocab_size=len(tel_vocab)+2 #this +2 will help the model for the additional special tokens
        outputs=torch.zeros(tar_seq_length,batch_size,final_target_vocab_size).to(device=device)
        hidden=self.encoder.initialize_hidden()
        cell=self.encoder.initialize_cell()
        if(self.cell_type=="RNN" or self.cell_type=="GRU"):
            encoder_output,hidden=self.encoder(input_seq,hidden,cell)
        else:
            encoder_output,hidden,cell=self.encoder(input_seq,hidden,cell)
        if(self.decoder_layers!=self.encoder_layers or self.bidirectional_bit):
            if(self.cell_type=="RNN" or self.cell_type=="GRU" or self.cell_type=="LSTM"):
                hidden=hidden[self.encoder_layers-1]+hidden[self.encoder_layers-1]
                hidden=hidden.repeat(self.decoder_layers,1,1)
            if(self.cell_type=="LSTM"):
                cell=cell[self.encoder_layers-1]+cell[self.encoder_layers-1]
                cell=cell.repeat(self.decoder_layers,1,1)
        x=target[0]
        for t in range(1,tar_seq_length):
            if(self.cell_type=="RNN" or self.cell_type=="GRU"):
                output,hidden=self.decoder(x,encoder_output,hidden)
            else:
                output,hidden,cell=self.decoder(x,encoder_output,hidden,cell)
            outputs[t]=output
            predicted=output.argmax(1)
            if(random.random()<teacher_force_ratio):
                x=target[t]
            else:
                x=predicted
        return outputs
    

def recurrent_neural_network(cell_type, bi_directional_bit, embedding_size, enc_dropout, dec_dropout, enc_layers, dec_layers, hidden_size, batch_size, attention, learning_rate, max_epochs,attention_bit):
    enc_input_size=len(eng_vocab)+2
    dec_input_size=len(tel_vocab)+2
    output_size=len(tel_vocab)+2
    # encoder network decleration 
    encoder_section=Encoder(enc_input_size,embedding_size,enc_layers,hidden_size,cell_type,bi_directional_bit,enc_dropout,batch_size).to(device=device)
    # decoder network decleration
    if(attention_bit):
        decoder_section=attention_add_decoder(dec_input_size,embedding_size,hidden_size,output_size,cell_type,dec_layers,dec_dropout,bi_directional_bit).to(device=device)
    else:
        decoder_section=Decoder(dec_input_size,embedding_size,hidden_size,dec_layers,dec_dropout,cell_type,output_size).to(device=device)
    model=Seq_to_seq(decoder_section,encoder_section,cell_type,bi_directional_bit,enc_layers,dec_layers)
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    pad = len(tel_vocab)+1
    loss_criterion=nn.CrossEntropyLoss(ignore_index=pad)
#     print(batch_size)
    for itr in range(max_epochs):
        print("epoch: ",itr+1)
        
        model.train()
        final_loss=0
        step=0
        for batch_id in tqdm(range((int)(len(eng_matrix_train)/batch_size))):
            inp_word=eng_matrix_train[batch_size*batch_id:batch_size*(batch_id+1)].to(device=device)
            out_word=tel_matrix_train[batch_size*batch_id:batch_size*(batch_id+1)].to(device=device)
            out_word=out_word.T
            inp_word=inp_word.T
            output=model(inp_word,out_word)
            
            output=output[1:].reshape(-1,output.shape[2])
            out_word=out_word[1:].reshape(-1)
            optimizer.zero_grad()
            loss=loss_criterion(output,out_word)
            final_loss=final_loss+loss
            loss.backward()
            
            torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=1)
            optimizer.step()
            step=step+1
        print("total loss: ",final_loss/step)
        train_acc=accuracy_fun(eng_matrix_train,tel_matrix_train,batch_size,max_epochs,model)
        valid_acc=accuracy_fun(eng_matrix_valid,tel_matrix_valid,batch_size,max_epochs,model)
        test_acc=accuracy_fun(eng_matrix_test,tel_matrix_test,batch_size,max_epochs,model)
        print("train accuracy: ",train_acc)
        print("valid accuracy: ",valid_acc)
        print("test accuracy: ",test_acc)
    # returned_pred=vectors_to_actual_words(model,eng_matrix_test,tel_matrix_test,'Test')
    # csv_file_func(returned_pred)
    # print("csv created")
#         wandb.log({'training_accuracy':train_acc})
#         wandb.log({'valid_accuracy':valid_acc})
#         wandb.log({'epochs':itr+1})
#     print(vectors_to_actual_words(model,eng_matrix_train,tel_matrix_train,batch_size,"Train"))
#     returned=vectors_to_actual_words(model,eng_matrix_valid,tel_matrix_valid,batch_size,"Valid")
#     for i in range(len(returned)):
#         print(returned[i])



def accuracy_fun(eng_matrix,tel_matrix,batch_size,max_epochs,model):
    correct=0
#     print(batch_size)
#     print(len(eng_matrix))
    for batch_id in range((int)(len(eng_matrix)/batch_size)):
        inp_word=eng_matrix[batch_size*batch_id:batch_size*(batch_id+1)].to(device=device)
        out_word=tel_matrix[batch_size*(batch_id):batch_size*(batch_id+1)].to(device=device)
        inp_word=inp_word.T
        out_word=out_word.T
        output=model.forward(inp_word,out_word,0)
        output=nn.Softmax(dim=2)(output)
        output=torch.argmax(output,dim=2)
#         print(batch_id)
#         print(batch_id,output)
        output=output.T
#         print(output)
        out_word=out_word.T
        for i in range(batch_size):
#             print(output[batch_id*batch_size+i],tel_list_train[batch_id*batch_size+i])
#             print(output[i],out_word[i])
            if(torch.equal(output[i][1:],out_word[i][1:])):
#                 print(output[i],out_word[i])
                correct=correct+1
#                 print('hello',output[i][1:],out_word[i][1:])
    return ((correct*100)/len(eng_matrix))



#convertion of the vectors of words to actual words back
# pred_train=[]
# pred_valid=[]
# pred_test_attn=[]
# def vectors_to_actual_words(model,english_mat,telugu_mat,batch_size,type_of_data):
#     for batch_id in range((int)(len(english_mat)/batch_size)):
#         input_batch=english_mat[batch_id*batch_size:batch_size*(batch_id+1)].to(device=device)
#         output_batch=telugu_mat[batch_id*batch_size:batch_size*(batch_id+1)].to(device=device)
#         output=model.forward(input_batch.T,output_batch.T,0)
#         output=nn.Softmax(dim=2)(output)
#         output=torch.argmax(output,dim=2)
#         output=output.T
#         for ind in range(len(output_batch)):
#             res_word=output_batch[ind]
#             pred_word=output[ind]
#             word_res=""
#             word_pred=""
#             inp_word=""
#             for i in range(len(pred_word)):
#                 if(pred_word[i]>0 and pred_word[i]<len(tel_vocab)):
#                     word_pred+=tel_vocab[pred_word[i]-1]
#             input_actual=input_batch[ind]
#             for i in range(len(input_actual)):
#                 if(input_actual[i]>0 and input_actual[i]<len(eng_vocab)):
#                     inp_word+=eng_vocab[input_actual[i]-1]
#             for i in range(len(res_word)):
#                 if(res_word[i]>0 and res_word[i]<len(tel_vocab)):
#                     word_res+=tel_vocab[res_word[i]-1]
#             temp=[inp_word,word_pred,word_res]
#             if(type_of_data=='Test'):
#                 pred_test_attn.append(temp)



def argument_parsing():
    rec_net=argparse.ArgumentParser(description='Training Model')
    rec_net.add_argument('-wp','--wandb_project',type=str,default='DL_assignment_3')
    rec_net.add_argument('-e_lay','--enc_layers',type=int,default=3)
    rec_net.add_argument('-bi','--bi_directional_bit',type=bool,default=True)
    rec_net.add_argument('-d_lay','--dec_layers',type=int,default=3)
    rec_net.add_argument('-b_size','--batch_size',type=int,default=512)
    rec_net.add_argument('-emd_size','--embedding_size',type=int,default=256)
    rec_net.add_argument('-h_size','--hidden_size',type=int,default=512)
    rec_net.add_argument('-e_drp','--enc_dropout',type=float,default=0)
    rec_net.add_argument('-d_drp','--dec_dropout',type=float,default=0)
    rec_net.add_argument('-epoch','--max_epochs',type=int,default=30)
    rec_net.add_argument('-lr','--learning_rate',type=float,default=1e-3)
    rec_net.add_argument('-ct','--cell_type',type=str,default='GRU')
    rec_net.add_argument('-atn','--attention_bit',type=bool,default=True)
    return rec_net.parse_args()

parser=argument_parsing()
recurrent_neural_network(parser.cell_type,parser.bi_directional_bit,parser.embedding_size,parser.enc_dropout,parser.dec_dropout,parser.enc_layers,parser.dec_layers,parser.hidden_size,parser.batch_size,parser.attention_bit,parser.learning_rate,parser.max_epochs,parser.attention_bit)


