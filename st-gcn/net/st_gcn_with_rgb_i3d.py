import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.models import i3d
from net.st_gcn import st_gcn
from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from net.heads.regressor import Evaluator



class Regressor(nn.Module):


    def __init__(self,output_dim,model_type='single',num_judge=1,dropout_ratio=0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio

        self.layer_s = nn.Linear(256,128)
        self.layer_v = nn.Linear(512,128)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.final = Evaluator(256,output_dim,model_type=model_type,num_judges=num_judge,dropout_ratio=dropout_ratio)
        


        
        
    def forward(self, feature_s, feature_v):
        # feature skeleton
        x_s = self.layer_s(feature_s)
        # feature RGB video
        x_v = self.avg_pool(feature_v)
        N, _, _, _, _ = x_v.size()

        x_v = x_v.view(N,-1)
        x_v = self.layer_v(x_v)

        # print(f"reg_s:{x_s.shape}")
        # print(f"reg_v:{x_v.shape}")
        
        out = torch.cat((x_s,x_v),dim=1)
        # print(out.shape)
        del x_v
        del x_s
        
        out = self.final(out)
        
        return out

        

class Model(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting,dropout=0.,num_judge=1,model_type='single', **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 2, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 2, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # feature extractor for video clip
        self.i3d =i3d.load_backbone("resnet18")
        
        # regression for assessment
        self.reg =  Regressor(num_class,model_type=model_type,num_judge=num_judge,dropout_ratio=dropout)

    def forward(self, x, x_v):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        x = x.view(N,-1)
        
        # RGB clip feature extracting
        feature_v = self.i3d(x_v)
        # prediction
        
        x = self.reg(x,feature_v)
        del feature_v
        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        # x = self.fcn(x)
        # output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        output = self.reg(x)
        return output, feature

