"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_for_sequence(input, conv):
    # assuming the input shapes are (B, S, C, H, W)
    # view will preserve the order when reshape back
    shape = input.shape
    _input = input.view(-1, shape[2], shape[3], shape[4])
    output = conv(_input)
    return output.reshape(shape[0], output.shape[0] // shape[0]
                          , output.shape[1], output.shape[2], output.shape[3])

class MultiHeadAttention2D(nn.Module):
    def __init__(self, q_in_channels, q_out_channels, q_kernel, q_stride, q_padding
                 , k_in_channels, k_out_channels, k_kernel, k_stride, k_padding
                 , v_in_channels, v_out_channels, v_kernel, v_stride, v_padding
                 , transform_kernel, transform_stride, transform_padding):
        super().__init__()
        self.q_conv = nn.Conv2d(q_in_channels, q_out_channels, q_kernel, q_stride, q_padding)
        self.k_conv = nn.Conv2d(k_in_channels, k_out_channels, k_kernel, k_stride, k_padding)
        self.v_conv = nn.Conv2d(v_in_channels, v_out_channels, v_kernel, v_stride, v_padding)
        self.transform_conv = nn.Conv2d(q_out_channels + k_out_channels, v_out_channels
                                        , transform_kernel, transform_stride, transform_padding)
        self.softmax = nn.Softmax(dim=1)



    def forward(self, q, k, v):
        #assuming the input shapes are (B, S, C, H, W)
        #the input is transform the single image into multiple attention
        #we still need the 2d conv layer for filtering so we need to use view method
        _q = F.relu(conv_for_sequence(q, self.q_conv), inplace=True)
        _k = F.relu(conv_for_sequence(k, self.k_conv), inplace=True)
        _v = F.relu(conv_for_sequence(v, self.v_conv), inplace=True)

        outputs = []

        #the outputs tensor is still (B, S, C, H, W)
        #the sequence length of output is the same as query's
        #one query makes one output
        #query can also be treated as the pre state in the RNN
        for i in range(_q.shape[1]):
            #only pick the one step of query once a time
            qi = _q[:, i:(i+1), :, :, :]
            qi = qi.expand((-1, _k.shape[1], -1, -1, -1))
            #assuming the tensor shape is (B, S, C, H, W)
            #concatenate on the channel dimension
            weight = torch.cat((qi, _k), dim=2)
            weight = conv_for_sequence(weight, self.transform_conv)
            #the softmax will make sequence dimension sum to 1
            #this defined in the constructor
            weight = self.softmax(weight)
            #we weighted sum all the values into on step in the sequence
            #keep the rest dimensions unchanged
            vi = torch.mul(_v, weight).sum(dim=1, keepdim=True)
            outputs.append(vi)

        #concatenate the tensor on the sequence dimension
        return torch.cat(outputs, dim=1)

