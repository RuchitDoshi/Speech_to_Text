import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Dataset 
import torch.nn.utils.rnn as rnn
from torch.nn.utils.rnn import *
import sys
import time
import os
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Cuda=torch.cuda.get_device_name(0)
print(Cuda)


#Change the path of the data files
speech_train = np.load('train_new.npy', allow_pickle=True, encoding='bytes')
speech_valid = np.load('dev_new.npy', allow_pickle=True, encoding='bytes')
speech_test = np.load('test_new.npy', allow_pickle=True, encoding='bytes')

transcript_train = np.load('train_transcripts.npy', allow_pickle=True,encoding='bytes')
transcript_valid = np.load('dev_transcripts.npy', allow_pickle=True,encoding='bytes')

LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', \
               'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']


def transform_letter_to_index(transcript, letter_list):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    output=[]
    for item in range(transcript.shape[0]):
        temp=''
        for i in transcript[item]:
            i=i.decode('UTF-8')
            for j in i:
                temp+=j
            temp+=' '
        temp=temp[:-1]
        trans=[len(letter_list)-2]
        for i in temp:
            trans.append(letter_list.index(i))
        trans.append(len(letter_list)-1)
        output.append(np.array(trans).astype(int))
    return np.array(output)

def indices_to_string(file):
    file = file.detach().cpu().numpy()
    sub = []
    for i in file:
        string = ''
        for j in i:
            string+=LETTER_LIST[j]
        sub.append(string)
    return sub

character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)


#Data Loader
class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train(batch_data):
    ### Return the padded speech and text data, and the length of utterance and transcript ###
    X,Y= zip(*batch_data)
    
    X_lens = torch.tensor([len(seq) for seq in X])
    Y_lens = torch.tensor([len(seq) for seq in Y])
    
    X=pad_sequence(X,batch_first=True,padding_value=0)
    Y=pad_sequence(Y,batch_first=True,padding_value=0)
    
    return X,Y,X_lens,Y_lens


def collate_test(batch_data):
    ### Return padded speech and length of utterance ###
    X=batch_data
    X_lens = torch.tensor([len(seq) for seq in X])
    
    X=pad_sequence(X,batch_first=True,padding_value=0)
        
    
    return X,X_lens

#Models:
class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.hidden_size=hidden_dim
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True,batch_first=True)

    def forward(self, x):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T/2, 2*H) encoded sequence from pyramidal Bi-LSTM 
        '''
        out,_=self.blstm(x)
        
        out,lens=pad_packed_sequence(out,batch_first=True)
        
        if out.shape[1]%2!=0:
            out=out[:,:-1,:]
        
        new_lens=np.zeros((lens.shape))
        for i in range(lens.shape[0]):
            if lens[i]%2!=0:
                lens[i]-=1
            new_lens[i]=int(lens[i]/2)
        
        new_lens=torch.tensor(new_lens.astype(int))
        assert (out.shape[1]%2==0)
    
        out=out.contiguous().view(out.shape[0],int(out.shape[1]/2),2,out.shape[2])#issue
        out=torch.max(out,2)[0]
        
        
        out = pack_padded_sequence(out, lengths=new_lens, batch_first=True, enforce_sorted=False)
        return out


class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''
    def __init__(self, input_dim, hidden_size, value_size=256,key_size=256):
        super(Encoder, self).__init__()
        self.cnn=nn.Conv1d(40,128,kernel_size=3,stride=1,padding=1)
        self.bnn=nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=1, bidirectional=True,batch_first=True)
        
        ### Add code to define the blocks of pBLSTMs! ###
        self.pblstm1=pBLSTM(2*hidden_size,hidden_size)
        self.pblstm2=pBLSTM(2*hidden_size,hidden_size)
        self.pblstm3=pBLSTM(2*hidden_size,hidden_size)

        self.key_network = nn.Linear(hidden_size*2, value_size)
        self.value_network = nn.Linear(hidden_size*2, key_size)

    def forward(self, x, lens):
        x=x.permute(0,2,1)
        x=F.leaky_relu_(self.bnn(self.cnn(x)))
        x=x.permute(0,2,1)
        rnn_inp = pack_padded_sequence(x, lengths=lens, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)
        

        ### Use the outputs and pass it through the pBLSTM blocks! ###
        
        outputs=self.pblstm1(outputs)
        outputs=self.pblstm2(outputs)
        outputs=self.pblstm3(outputs)

        linear_input, _ = pad_packed_sequence(outputs,batch_first=True)

        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value

class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value,lens):
        '''
        :param query :(N, context_size) Query is the output of LSTMCell from Decoder
        :param key: (N, key_size) Key Projection from Encoder per time step
        :param value: (N, value_size) Value Projection from Encoder per time step
        :return output: Attended Context
        :return attention_mask: Attention mask that can be plotted  
        '''
        energy = torch.bmm(key, query.unsqueeze(2))

        energy=energy.squeeze(-1)

        mask = torch.arange(energy.size(1)).to(device).unsqueeze(0) >= lens.unsqueeze(1)
        
        energy.masked_fill_(mask, -1e9)

        energy=energy.unsqueeze(2)
        
        attention = nn.functional.softmax(energy, dim=1)

        context = torch.bmm(attention.permute(0,2,1), value)
        
        return context.squeeze(1), attention.squeeze(2)


class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step, 
    thus we use LSTMCell instead of LSTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
   

    def __init__(self, vocab_size, hidden_dim, value_size=256, key_size=256, isAttended=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=hidden_dim)
        self.lstm2 = nn.LSTMCell(input_size=hidden_dim, hidden_size=key_size)

        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention().to(device)

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)

    def forward(self, key, values, text,lens,teach,isTrain=True):
        '''
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''
        batch_size = key.shape[0]

        if (isTrain == True):
            embeddings = self.embedding(text.long())
            max_len =text.shape[1]-1
            idx=np.random.permutation(max_len)
            idx=idx[:int(max_len*teach)]
        else:
            max_len=250
            idx=[]
            
        lens=lens//8

        context= torch.zeros((key.shape[0],key.shape[2])).to(device)
        predictions = []
        hidden_states = [None, None]
        prediction = (torch.zeros(batch_size,35)).to(device)
        prediction[:,33]=1

        for i in range(max_len):
            # * Implement Gumble noise and teacher forcing techniques 
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break 
            #   out of the loop so you do you do not get index out of range errors. 

            
            if (isTrain) and i not in idx:
                char_embed = embeddings[:,i,:]
            else:         
                char_embed = self.embedding(prediction.argmax(dim=-1))
          

            inp = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell ###
            output = hidden_states[1][0]
            
            context,_=self.attention(output,key,values,lens)

            prediction = self.character_prob(torch.cat([output, context], dim=1))
            
            predictions.append(prediction.unsqueeze(1))

        return torch.cat(predictions, dim=1)

class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=256, key_size=256, isAttended=False):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, hidden_dim)

    def forward(self, speech_input, speech_len,text_input,teach, isTrain=True,):
        key, value = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(key, value,text_input,speech_len,teach)
        else:
            predictions = self.decoder(key, value,None,speech_len,teach,isTrain=False)
        return predictions


def masked_loss (loss, lengths):
    # mask = torch.zeros(loss.shape).to(device)
    mask = torch.arange(loss.size(1)).to(device).unsqueeze(0) >= lengths.unsqueeze(1)
    loss.masked_fill_(mask, 0)
    temp=torch.sum(loss)/torch.sum(lengths)
    return temp


#Train function:
def train(model, train_loader, criterion, optimizer,teach):
    model.train()
    train_loss=[]
    perplexity=[]
    for batch_num,(X,Y,X_lens,Y_lens) in enumerate(train_loader):
        if batch_num%50==0:
            print('Batch_num: ',batch_num)
        X,Y,X_lens,Y_lens=X.to(device),Y.to(device),X_lens.to(device),Y_lens.to(device)
        optimizer.zero_grad()
        out = model(X, X_lens,Y,teach)
        pred=out.argmax(-1)
        pred=indices_to_string(pred)
        Y_temp=Y[:,1:]
        loss = criterion(out.view(-1,out.size(2)),Y_temp.flatten().long())
        loss = loss.reshape(out.shape[0], out.shape[1])
        m_loss = masked_loss(loss, Y_lens)
        perplexity.append(np.exp(m_loss.detach().cpu().numpy()))
        m_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        train_loss.append(m_loss.item())
        del X,X_lens,Y,Y_lens,loss
        torch.cuda.empty_cache()
    print('Perplexity: ',sum(perplexity)/len(perplexity))
    print('Train_Loss',sum(train_loss)/len(train_loss))
    print('Train example: ', pred[0])


#Validation Function
def validation(model,val_loader,criterion, optimizer,teach):
    model.eval()
    dev_predictions=[]
    for batch_num,(X,Y,X_lens,Y_lens) in enumerate(val_loader):
        X,Y,X_lens,Y_lens=X.to(device),Y.to(device),X_lens.to(device),Y_lens.to(device)
        out = model(X, X_lens,None,teach,False)
        out=out.argmax(-1)
        pred=indices_to_string(out)
        for i in pred:
            dev_predictions.append(i)
    return dev_predictions


#Test Functions
def test(model,test_loader,criterion,optimizer,teach):
    model.eval()
    test_predictions=[]
    for batch_num,(X,X_lens) in enumerate(test_loader):
        X,X_lens=X.to(device),X_lens.to(device)
        out=model(X,X_lens,None,teach,False)
        out=out.argmax(-1)
        pred=indices_to_string(out)
        for i in pred:
            test_predictions.append(i)
    return test_predictions


#Main Function
model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=512)
model=model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(reduction='none')
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,40,50,60], gamma=0.1)


nepochs = 40
batch_size = 128 if device == 'cuda' else 1

# speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)

train_dataset = Speech2TextDataset(speech_train, character_text_train)
validation_dataset=Speech2TextDataset(speech_valid, character_text_valid)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_train)


test_dataset = Speech2TextDataset(speech_test, None, False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_test)
print('Data Loaded')

TRAIN=True
if TRAIN:
    teach=0.1
    for epoch in range(nepochs):
        print('Epoch: ',epoch)
        print('Teacher Forcing: ',teach)
        start_time=time.time()
        train(model, train_loader, criterion, optimizer,teach)
        dev_predictions=validation(model,val_loader,criterion, optimizer,teach)
        test_predictions=test(model,test_loader,criterion,optimizer,teach)
        scheduler.step(epoch)
        if epoch%4==0 and teach <0.5:
            teach+=0.03
        print('Time : ',time.time()-start_time)
        print('Validation_Prediction: ',dev_predictions[0])
        print('Test_prediction: ',test_predictions[0])
        path='Weights-'+str(epoch+1)+'.pth'
        torch.save(model.state_dict(),path)


#Post Processing
TEST=False
if TEST:
    Path='' #load the weights with the best results on train, test and validation dataset
    model.load_state_dict(torch.load(Path))
    print(optimizer)
    test_predictions=test(model,test_loader,criterion,optimizer,teach)

    predictions=[]
    for i in range(len(test_predictions[0])):
        temp=''
        for j in range(len(test_predictions[i])):
            if (test_predictions[i][j])=='<':
                break
            temp+=test_predictions[i][j]
        predictions.append(temp)
    
    name='Submission.csv'
    f=open(name,'w')
    f.write('ID')
    f.write(',')
    f.write('Predicted')
    f.write('\n')
    for i in range(len(predictions)):
        f.write(str(i))
        f.write(',')
        f.write(predictions[i])
        f.write('\n')
    f.close()