import torch
import math
from torch import nn
from torchsummary import summary


class Tinyrcnn(nn.Module):
    def __init__(self,device):
        super(Tinyrcnn, self).__init__()
        self.device = device

        self.flatten = nn.Flatten()

        self.block_1 = nn.Sequential(
             nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(5,4)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,1), stride=(1,2))
                )
        
        self.block_2 = nn.Sequential(
             nn.Conv2d(
                in_channels=32,
                out_channels=48,
                kernel_size=(3,4)),
                nn.Dropout(p=0.2),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,1), stride=(1,3))
                
                )
        
        self.block_3 = nn.Sequential(
             nn.Conv2d(
                in_channels=48,
                out_channels=128,
                kernel_size=(4,4)),
                nn.Dropout(p=0.2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,1), stride=(1,2))
                
                )
        
        
        #for mel spectogram
        self.gru = nn.GRU(input_size=414, hidden_size=64 ,
                           num_layers= 1,  batch_first= True )

        self.linear = nn.Linear(in_features=128*64, out_features=1)
#---------------------------------------------------------------------------------#

        #for MFCC
        # self.gru = nn.GRU(input_size=564, hidden_size=64 ,
        #                    num_layers= 1,  batch_first= True )
        
        
        # self.linear = nn.Linear(in_features=128*64, out_features=1)

      
       
        
    def _scaled_dot_product_attention(self, query, key, value,
                                         attn_mask=None, dropout_p=0.0,
                                         is_causal=False, scale=None) -> torch.Tensor:
            # Ensure tensors are on the same device as the model

            # query, key, value = query.to(self.device), key.to(self.device), value.to(self.device)
            L, S = query.size(-2), key.size(-2)
            scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
            attn_bias = torch.zeros(L, S, dtype=query.dtype  )
            if is_causal:
                assert attn_mask is None
                temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                attn_bias.to(query.dtype)
        
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
                else:
                    attn_bias += attn_mask
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight += attn_bias.to(self.device)
            attn_weight = torch.softmax(attn_weight, dim=-1)
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            return attn_weight @ value
        
    def forward(self , input):
        

            x = self.block_1(input)
            x = self.block_2(x)
            x= self.block_3(x)
            # Reshape the input tensor to have the sequence dimension as the second dimension
            batch_size, channels, height, width = x.size()
            # batch_size, channels, _, _ = x.size()
            x = x.view(batch_size, channels, -1) 
            # print(x.shape)
            output, _ = self.gru(x)
   
            attention_weights = self._scaled_dot_product_attention(output,output,output)
            # print(f'attention_weights {attention_weights.shape}')
            # 
            # output = attention_weights.view(attention_weights.size(0), -1)

            output = self.flatten(attention_weights)

            # print(f'flatten {output.shape}')
            output = self.linear(output)
            # print(f'linear output - {output.shape}')
            return output
    

if __name__=="__main__":
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     model = Tinyrcnn(device=device).to(device)
    #  a = torch.rand(32,1,100,79).to(device)
    #  a = torch.rand(1,1,100,79).to(device) #for mel spectogram
     a = torch.rand(1,1,200,59).to(device) #for MFCC

     res = model(a)
     print(res)



