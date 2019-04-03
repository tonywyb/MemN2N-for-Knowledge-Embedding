# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 23:44:36 2019

@author: Lenovo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json


learning_rate = 0.01
data_path = "train&test.json"
en_path = "default"
de_path = "default"
se_path = "default"
train_mode = 1
SOS = 1
EOS = 0
epoch = 20000
dataset = {}


class WIdic:
    def __init__(self):   
        self.word2index = {"<EOS>": 0, "START": 1}
        self.index2word = {0: "<EOS>", 1: "START"}
        self.totalwords = 2
        
    def addword(self, word):
        if self.word2index.get(word) == None:    
            self.word2index[word] = self.totalwords
            self.index2word[self.totalwords] = word
            self.totalwords = self.totalwords + 1
    
    def addsent(self, sent):
        for word in sent.split():
            self.addword(word)
    
    def W2I(self, sent):
        outv = []
        for words in sent.split():
            outv.append(self.word2index[words])
        outv.append(0)
        return outv
            
    def I2W(self, inv):
        outs = ""
        for index in inv:
            outs = outs + self.index2word[index]
        return outs


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.add_module("LSTM", nn.LSTM(input_size=hidden_s, hidden_size=hidden_s, batch_first=False))
        
    def forward(self, embedded_seq, hidden):
        embedded_seq = embedded_seq.view(1, 1, -1)
        outputs, hidden = self.LSTM(embedded_seq, hidden)
        outputs = outputs[0][0]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.add_module("LSTM", nn.LSTM(input_size=hidden_s, hidden_size=hidden_s, batch_first=False))
        self.add_module("linear_out", nn.Linear(hidden_s, output_s))
        
    def forward(self, embedded_seq, hidden=None):
        embedded_seq = embedded_seq.view(1, 1, -1)
        outputs, hidden = self.LSTM(embedded_seq, hidden)
        out = self.linear_out(outputs[0])
        out = F.log_softmax(out, dim=1)
        return out[0], hidden

    
class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()
        self.add_module("embedding", nn.Embedding(input_s, hidden_s))
        
    def forward(self, input_seq, output_seq, encoder, decoder, hidden, train_mode, lossfunc):
        input_length = input_seq.size(0)
        output_length = output_seq.size(0)
        embedded_in = self.embedding(input_seq)
        embedded_out = self.embedding(output_seq)
        encoder_outputs = None
        encoder_hidden = hidden
        for ei in range(input_length):
            encoder_outputs, encoder_hidden = encoder.forward(embedded_in[ei], encoder_hidden)
        decoder_hidden = encoder_hidden
        decoder_outputs = torch.zeros(output_length, output_s)
        decoder_input = self.embedding(torch.tensor(SOS))
        if train_mode == 1:
            for di in range(output_length):
                decoder_outputs[di], decoder_hidden = decoder.forward(decoder_input, decoder_hidden)    
                decoder_input = embedded_out[di]
        else:
            for di in range(output_length):
                decoder_outputs[di], decoder_hidden = decoder.forward(embedded_out[di], decoder_hidden)
                max_pro, max_index = torch.topk(decoder_outputs[di], 1)
                decoder_in = max_index.squeeze()
                if decoder_in == EOS:
                    break
                decoder_input = self.embedding(decoder_in)
        return decoder_outputs, decoder_hidden


mydic = WIdic()
data = open(data_path, 'r', encoding='utf-8')
dataset = json.load(data)
data.close()
train_num = len(dataset["train"])
test_num = len(dataset["test"])


def get_sent(prob):
    maxp, maxi = torch.topk(prob, 1)
    outt = list(maxi.squeeze())
    outv = []
    for i in outt:
        outv.append(i.item())
    outs = mydic.I2W(outv)
    return outs
    

def train(text, seq2seq, encoder, decoder, train_mode, lossfunc, encoder_optimizer, decoder_optimizer, seq2seq_optimizer):
    total_lines = int(len(text) / 2) - 1
    loss = 0
    hidden = (torch.zeros(1, 1, hidden_s), torch.zeros(1, 1, hidden_s))
    for li in range(total_lines):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        seq2seq_optimizer.zero_grad()
        if li == 0:
            in_sent = text[0] + " " + text[1]
            out_sent = text[2]
        else:
            in_sent = text[2 * li + 1]
            out_sent = text[2 * li + 2]
        print("input: " + in_sent)
        print("wanted: " + out_sent)
        in_seq = torch.LongTensor(mydic.W2I(in_sent))
        out_seq = torch.LongTensor(mydic.W2I(out_sent))
        decoder_outputs, hidden = seq2seq.forward(in_seq, out_seq, encoder, decoder, hidden, train_mode, lossfunc)
        temp_loss = lossfunc(decoder_outputs, out_seq)
        temp_loss.backward(retain_graph=True)
        encoder_optimizer.step()
        decoder_optimizer.step()
        seq2seq_optimizer.step()
        loss += float(temp_loss)
        predicted = get_sent(decoder_outputs)
        print("Bot's output: " + predicted)
    
    loss = loss / total_lines
    print("average loss on the text: ", float(loss))


def init():
    for tri in dataset["train"]:
        for sent in dataset["train"][tri]["conversation"]:
            mydic.addsent(sent)
    for tei in dataset["test"]:
        for sent in dataset["test"][tei]["history"]:
            mydic.addsent(sent)
        g_sent = ""
        for gi in dataset["test"][tei]["goal"]:
            g_sent = gi[0] + " " + gi[1] + " " + gi[2]
        mydic.addsent(g_sent)
    print("data prepared")
 
    
def run():
    myencoder = Encoder()
    mydecoder = Decoder()
    myseq2seq = Seq2Seq()
    if train_mode == 0:
        myencoder = torch.load(en_path)
        myencoder.eval()
        mydecoder = torch.load(de_path)
        mydecoder.eval()
        myseq2seq = torch.load(se_path)
        myseq2seq.eval()
    en_optim = torch.optim.SGD(myencoder.parameters(), lr = learning_rate, momentum = 0.5, dampening = 0.1)
    de_optim = torch.optim.SGD(mydecoder.parameters(), lr = learning_rate, momentum = 0.5, dampening = 0.1)
    se_optim = torch.optim.SGD(myseq2seq.parameters(), lr = learning_rate, momentum = 0.5, dampening = 0.1)
    lossfunc = nn.NLLLoss()
    for epi in range(epoch):
        print(epi, ":")
        if train_mode == 1:
            chosen = random.randint(0, train_num - 1)
            text = dataset["train"][str(chosen)]["conversation"]
        else:
            chosen = random.randint(0, test_num - 1)
            text = dataset["test"][str(chosen)]["conversation"]
        train(text, myseq2seq, myencoder, mydecoder, train_mode, lossfunc, en_optim, de_optim, se_optim)
        if train_mode == 1 and (epi + 1) % 5000 == 0:
            torch.save(myencoder, "encoder_epoch_" + str(epi) + ".pt")
            torch.save(mydecoder, "decoder_epoch_" + str(epi) + ".pt")
            torch.save(myseq2seq, "seq2seq_epoch_" + str(epi) + ".pt")
   
     
if __name__ == "__main__":
    init()
    input_s = mydic.totalwords
    hidden_s = 512
    output_s = input_s
    run()