import torch.nn as nn


class MLP(nn.Module):
    def __init__(
            self,
            inp_dim,            # Input dimension.
            latent_dim,         # Hidden layer dimension.
            out_dim,            # Output dimension.
            num_layers=2,       # Number of layers (incl. input & output).
            bias=True,          # Bias term in Linear layers.
            batchnorm=True,     # Use BatchNorm.
            layernorm=False,    # Use LayerNorm.
            dropout=0,          
            end_relu=False,     # Use ReLU at the end.
            drop_input=0,       # Dropout at input.
            drop_output=0,       # Dropout at output.
            final_linear_bias=True
        ):
        super(MLP, self).__init__()
        mod = []

        if drop_input > 0:
            mod.append(nn.Dropout(drop_input))

        mod.append(nn.Linear(inp_dim, latent_dim, bias=bias))
        if batchnorm:
            mod.append(nn.BatchNorm1d(latent_dim))
        if layernorm:
            mod.append(nn.LayerNorm(latent_dim))
        mod.append(nn.ReLU())

        for L in range(num_layers-2):
            mod.append(nn.Linear(latent_dim, latent_dim, bias=bias))
            if batchnorm:
                mod.append(nn.BatchNorm1d(latent_dim))
            if layernorm:
                mod.append(nn.LayerNorm(latent_dim))
            mod.append(nn.ReLU())
        
        if dropout > 0:
            mod.append(nn.Dropout(dropout))

        mod.append(nn.Linear(latent_dim, out_dim, bias=final_linear_bias))

        if end_relu:
            mod.append(nn.ReLU())

        if drop_output > 0:
            mod.append(nn.Dropout(drop_output))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output

class MLP2(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''
    def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers = []):
        super(MLP2, self).__init__()
        mod = []
        incoming = inp_dim
        for layer in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers.pop(0)
            mod.append(nn.Linear(incoming, outgoing, bias = bias))
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
            mod.append(nn.ReLU(inplace = True))
            if dropout:
                mod.append(nn.Dropout(p = 0.5))

        mod.append(nn.Linear(incoming, out_dim, bias = bias))

        if relu:
            mod.append(nn.ReLU(inplace = True))
        self.mod = nn.Sequential(*mod)
    
    def forward(self, x):
        return self.mod(x)
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x