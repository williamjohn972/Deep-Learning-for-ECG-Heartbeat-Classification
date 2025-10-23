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
class ECG_BiLSTM_Classifier(nn.Module):

    def __init__(self, 
                 input_size, 
                 hidden_size,
                 num_layers,
                 num_classes,
                 dropout=0.2
                ):
        
        super(ECG_BiLSTM_Classifier,self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_size * 2, out_features=num_classes)
    
    def forward(self,x):
        # current x: [batch, seq_len] --> we need to add an extra dim for num_features per timestamp
        # since batch_first = True, x: [batch, seq_len, num_features per timestep]
        x = x.unsqueeze(-1)

        out, (h_n, c_n) = self.lstm(x)
        out = out[:,-1,:] # extracting the output at the last time_step
        out = self.dropout(out)
        out = self.fc(out)
        return out 
    




