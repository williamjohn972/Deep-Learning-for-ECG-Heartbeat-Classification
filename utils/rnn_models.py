import torch 
from torch import nn

# Baseline RNN Model
class ECG_RNN_Classifier(nn.Module):

    def __init__(self, 
                 input_size, 
                 hidden_size,
                 num_layers,
                 num_classes,
                 dropout=0.0
                ):
        
        super(ECG_RNN_Classifier,self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=num_classes)
    
    def forward(self,x):
        # current x: [batch, seq_len] --> we need to add an extra dim for num_features per timestamp
        # since batch_first = True, x: [batch, seq_len, num_features per timestep]
        x = x.unsqueeze(-1)

        out, _ = self.rnn(x)
        out = out[:,-1,:] # extracting the output at the last time_step
        out = self.dropout(out)
        out = self.fc(out)
        return out 
    
# BaseLine LSTM Model 
class ECG_LSTM_Classifier(nn.Module):

    def __init__(self, 
                 input_size, 
                 hidden_size,
                 num_layers,
                 num_classes,
                 dropout=0.2,
                 bidirectional = False
                ):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1
        
        super(ECG_LSTM_Classifier,self).__init__()

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout= 0 if self.num_layers==1 else self.dropout,
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(
            in_features=self.hidden_size * self.num_directions, 
            out_features=self.num_classes)
    
    def forward(self,x):
        # current x: [batch, seq_len] --> we need to add an extra dim for num_features per timestamp
        
        # since batch_first = True, x should be : [batch, seq_len, num_features per timestep]
        x = x.unsqueeze(-1)

        # h_n shape --> [num_layers * num_directions, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            h_n = h_n.view(self.num_layers,self.num_directions,x.size(0),self.hidden_size)
            h_forward = h_n[-1,0,:,:]
            h_backward = h_n[-1,1,:,:]
            h_final = torch.cat((h_forward,h_backward), dim=1)

        else:
            h_final =  h_n[-1,:,:]  


        h_final = self.dropout_layer(h_final)
        h_final = self.linear_layer(h_final)
        return h_final
    
class ECG_GRU_Classifier(nn.Module):

    def __init__(self, 
                 input_size, 
                 hidden_size,
                 num_layers,
                 num_classes,
                 dropout=0.2,
                 bidirectional = False
                ):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.num_directions = 2 if bidirectional else 1
        
        super(ECG_GRU_Classifier,self).__init__()

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout= 0 if self.num_layers==1 else self.dropout,
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.linear_layer = nn.Linear(
            in_features=self.hidden_size * self.num_directions, 
            out_features=self.num_classes)
    
    def forward(self,x):
        # current x: [batch, seq_len] --> we need to add an extra dim for num_features per timestamp
        
        # since batch_first = True, x should be : [batch, seq_len, num_features per timestep]
        x = x.unsqueeze(-1)

        # h_n shape --> [num_layers * num_directions, batch, hidden_size]
        out, h_n = self.gru(x)

        if self.bidirectional:
            h_n = h_n.view(self.num_layers,self.num_directions,x.size(0),self.hidden_size)
            h_forward = h_n[-1,0,:,:]
            h_backward = h_n[-1,1,:,:]
            h_final = torch.cat((h_forward,h_backward), dim=1)

        else:
            h_final =  h_n[-1,:,:]  


        h_final = self.dropout_layer(h_final)
        h_final = self.linear_layer(h_final)
        return h_final

    

