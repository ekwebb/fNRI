"""
This code is based on https://github.com/ethanfetaya/NRI
(MIT licence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax

_EPS = 1e-10


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data) 
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class CNN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0,
                                 dilation=1, return_indices=False,
                                 ceil_mode=False)

        self.conv1 = nn.Conv1d(n_in, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(n_hid)
        self.conv2 = nn.Conv1d(n_hid, n_hid, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(n_hid)
        self.conv_predict = nn.Conv1d(n_hid, n_out, kernel_size=1)
        self.conv_attention = nn.Conv1d(n_hid, 1, kernel_size=1)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        # Input shape: [num_sims * num_edges, num_dims, num_timesteps]

        x = F.relu(self.conv1(inputs))
        x = self.bn1(x)
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        pred = self.conv_predict(x)
        attention = my_softmax(self.conv_attention(x), axis=2)

        edge_prob = (pred * attention).mean(dim=2)
        return edge_prob


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        self.factor = factor
        # n_hid = num edge types
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data) 
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)

        return self.fc_out(x)

class MLPEncoder_multi(nn.Module):
    def __init__(self, n_in, n_hid, edge_types_list, do_prob=0., split_point=1, 
                 init_type='xavier_normal', bias_init=0.0):
        super(MLPEncoder_multi, self).__init__()

        self.edge_types_list = edge_types_list
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        #print(self.mlp1.fc1.weight[0][0:5])
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)

        self.init_type = init_type
        if self.init_type not in [ 'xavier_normal', 'orthogonal', 'sparse' ]:
            raise ValueError('This initialization type has not been coded')
        #print('Using '+self.init_type+' for encoder weight initialization')
        self.bias_init = bias_init

        self.split_point = split_point
        if split_point == 0:
            self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            self.fc_out = nn.ModuleList([nn.Linear(n_hid, sum(edge_types_list))])
        elif split_point == 1:
            self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
            self.mlp4 = nn.ModuleList([MLP(n_hid * 3, n_hid, n_hid, do_prob) for _ in edge_types_list])
            self.fc_out =  nn.ModuleList([nn.Linear(n_hid, K) for K in edge_types_list])
        elif split_point == 2:
            self.mlp3 = nn.ModuleList([MLP(n_hid, n_hid, n_hid, do_prob) for _ in edge_types_list])
            self.mlp4 = nn.ModuleList([MLP(n_hid * 3, n_hid, n_hid, do_prob) for _ in edge_types_list])
            self.fc_out =  nn.ModuleList([nn.Linear(n_hid, K) for K in edge_types_list]) 
        else:
            raise ValueError('Split point is not valid, must be 0, 1, or 2')

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data)
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data)
                elif self.init_type == 'sparse':
                    nn.init.sparse_(m.weight.data, sparsity=0.1)

                if not math.isclose(self.bias_init, 0, rel_tol=1e-9):
                    m.bias.data.fill_(self.bias_init)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send)
        if self.split_point == 0:
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
            return self.fc_out[0](x)
        elif self.split_point == 1:
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            y_list = []
            for i in range(len(self.edge_types_list)):
                y = self.mlp4[i](x)
                y_list.append( self.fc_out[i](y) )
            return torch.cat(y_list,dim=-1)
        elif self.split_point == 2:
            y_list = []
            for i in range(len(self.edge_types_list)):
                y = self.mlp3[i](x)
                y = self.node2edge(y, rel_rec, rel_send)
                y = torch.cat((y, x_skip), dim=2)  # Skip connection
                y = self.mlp4[i](y)
                y_list.append( self.fc_out[i](y) )
            return torch.cat(y_list,dim=-1)

class MLPEncoder_sigmoid(nn.Module):
    def __init__(self, n_in, n_hid, num_factors, do_prob=0., split_point=1):
        super(MLPEncoder_sigmoid, self).__init__()

        self.num_factors = num_factors
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)

        self.split_point = split_point
        if split_point == 0:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            self.fc_out = nn.Linear(n_hid, num_factors)
        elif split_point == 1:
            self.mlp4 = nn.ModuleList([MLP(n_hid * 3, n_hid, n_hid, do_prob) for _ in range(num_factors)])
            self.fc_out =  nn.ModuleList([nn.Linear(n_hid, 1) for i in range(num_factors)])
        elif split_point == 2:
            self.mlp3 = nn.ModuleList([MLP(n_hid, n_hid, n_hid, do_prob) for _ in range(num_factors)])
            self.mlp4 = nn.ModuleList([MLP(n_hid * 3, n_hid, n_hid, do_prob) for _ in range(num_factors)])
            self.fc_out =  nn.ModuleList([nn.Linear(n_hid, 1) for i in range(num_factors)]) 
        else:
            raise ValueError('Split point is not valid, must be 0, 1, or 2')

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data) 
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send)
        if self.split_point == 0:
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
            return self.fc_out(x)
        elif self.split_point == 1:
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            y_list = []
            for i in range(self.num_factors):
                y = self.mlp4[i](x)
                y_list.append( self.fc_out[i](y) )
            return torch.cat(y_list,dim=-1)
        elif self.split_point == 2:
            y_list = []
            for i in range(self.num_factors):
                y = self.mlp3[i](x)
                y = self.node2edge(y, rel_rec, rel_send)
                y = torch.cat((y, x_skip), dim=2)  # Skip connection
                y = self.mlp4[i](y)
                y_list.append( self.fc_out[i](y) )
            return torch.cat(y_list,dim=-1)

class RandomEncoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, edge_types_list, cuda_on):
        super(RandomEncoder, self).__init__()

        self.edge_types_list = edge_types_list
        self.cuda_on = cuda_on
        print('Using a random encoder.')

    def forward(self, inputs, rel_rec, rel_send):
        n = inputs.shape[1]
        output = Variable(torch.randn(inputs.shape[0],n**2-n,sum(self.edge_types_list)))
        if self.cuda_on:
            output = output.cuda()
        return output

class OnesEncoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, edge_types_list, cuda_on):
        super(OnesEncoder, self).__init__()

        self.edge_types_list = edge_types_list
        self.cuda_on = cuda_on
        print('Using a "ones" encoder.')

    def forward(self, inputs, rel_rec, rel_send):
        n = inputs.shape[1]
        output = Variable(torch.ones(inputs.shape[0],n**2-n,sum(self.edge_types_list)))
        if self.cuda_on:
            output = output.cuda()
        return output


class CNNEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(CNNEncoder, self).__init__()
        self.dropout_prob = do_prob

        self.factor = factor

        self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
        self.fc_out = nn.Linear(n_hid, n_out)

        if self.factor:
            print("Using factor graph CNN encoder.")
        else:
            print("Using CNN encoder.")

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.view(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0) * receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0) * senders.size(1),
                               inputs.size(2),
                               inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([receivers, senders], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
        x = self.mlp1(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp2(x)

            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)

        return self.fc_out(x)

class CNNEncoder_multi(nn.Module):
    def __init__(self, n_in, n_hid, edge_types_list, do_prob=0., split_point=0, init_type='xavier_normal'):
        super(CNNEncoder_multi, self).__init__()
        self.dropout_prob = do_prob
  
        self.edge_types_list = edge_types_list

        self.init_type = init_type
        if self.init_type not in [ 'xavier_normal', 'orthogonal' ]:
            raise ValueError('This initialization type has not been coded')
        print('Using '+self.init_type+' for encoder weight initialization')

        self.cnn = CNN(n_in * 2, n_hid, n_hid, do_prob)
        self.mlp1 = MLP(n_hid, n_hid, n_hid, do_prob)

        self.split_point = split_point
        if split_point == 0:
            self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
            self.mlp3 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            self.fc_out = nn.Linear(n_hid, sum(edge_types_list))
        elif split_point == 1:
            self.mlp2 = MLP(n_hid, n_hid, n_hid, do_prob)
            self.mlp3 = nn.ModuleList([MLP(n_hid * 3, n_hid, n_hid, do_prob) for _ in edge_types_list])
            self.fc_out =  nn.ModuleList([nn.Linear(n_hid, K) for K in edge_types_list])
        elif split_point == 2:
            self.mlp2 = nn.ModuleList([MLP(n_hid, n_hid, n_hid, do_prob) for _ in edge_types_list])
            self.mlp3 = nn.ModuleList([MLP(n_hid * 3, n_hid, n_hid, do_prob) for _ in edge_types_list])
            self.fc_out =  nn.ModuleList([nn.Linear(n_hid, K) for K in edge_types_list]) 
        else:
            raise ValueError('Split point is not valid, must be 0, 1, or 2')

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data)
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data) 
                m.bias.data.fill_(0.1)

    def node2edge_temporal(self, inputs, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        x = inputs.view(inputs.size(0), inputs.size(1), -1)

        receivers = torch.matmul(rel_rec, x)
        receivers = receivers.view(inputs.size(0) * receivers.size(1),
                                   inputs.size(2), inputs.size(3))
        receivers = receivers.transpose(2, 1)

        senders = torch.matmul(rel_send, x)
        senders = senders.view(inputs.size(0) * senders.size(1),
                               inputs.size(2),
                               inputs.size(3))
        senders = senders.transpose(2, 1)

        # receivers and senders have shape:
        # [num_sims * num_edges, num_dims, num_timesteps]
        edges = torch.cat([receivers, senders], dim=1)
        return edges

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):

        # Input has shape: [num_sims, num_atoms, num_timesteps, num_dims]
        edges = self.node2edge_temporal(inputs, rel_rec, rel_send)
        x = self.cnn(edges)
        x = x.view(inputs.size(0), (inputs.size(1) - 1) * inputs.size(1), -1)
        x = self.mlp1(x)
        x_skip = x
        x = self.edge2node(x, rel_rec, rel_send)

        if self.split_point == 0:
            x = self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp3(x)
            return self.fc_out(x)
        elif self.split_point == 1:
            x = self.mlp2(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            y_list = []
            for i in range(len(self.edge_types_list)):
                y = self.mlp3[i](x)
                y_list.append( self.fc_out[i](y) )
            return torch.cat(y_list,dim=-1)
        elif self.split_point == 2:
            y_list = []
            for i in range(len(self.edge_types_list)):
                y = self.mlp2[i](x)
                y = self.node2edge(y, rel_rec, rel_send)
                y = torch.cat((y, x_skip), dim=2)  # Skip connection
                y = self.mlp3[i](y)
                y_list.append( self.fc_out[i](y) )
            return torch.cat(y_list,dim=-1)


class MLPDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False):
        super(MLPDecoder, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first_edge_type = skip_first

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
        else:
            start_idx = 0

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class MLPDecoder_multi(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, edge_types_list, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False, init_type='default'):
        super(MLPDecoder_multi, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first = skip_first
        self.edge_types = edge_types
        self.edge_types_list = edge_types_list

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

        self.init_type = init_type
        if self.init_type not in [ 'xavier_normal', 'orthogonal', 'default' ]:
            raise ValueError('This initialization type has not been coded')
        #print('Using '+self.init_type+' for decoder weight initialization')

        if self.init_type != 'default':
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data,gain=0.000001)
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data,gain=0.000001)
                #m.bias.data.fill_(0.1)

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        # non_null_idxs = list of indexs of edge types which as non null (i.e. edges over which messages can be passed)
        non_null_idxs = list(range(self.edge_types))
        if self.skip_first:
            # if skip_first is True, the first edge type in each factor block is null
            edge = 0
            for k in self.edge_types_list:
                non_null_idxs.remove(edge)
                edge += k

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in non_null_idxs:
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()

class MLPDecoder_sigmoid(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, num_factors, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False, init_type='default'):
        super(MLPDecoder_sigmoid, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(num_factors)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(num_factors)])
        self.msg_out_shape = msg_out
        self.num_factors = num_factors

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

        self.init_type = init_type
        if self.init_type not in [ 'xavier_normal', 'orthogonal', 'default' ]:
            raise ValueError('This initialization type has not been coded')
        #print('Using '+self.init_type+' for decoder weight initialization')

        if self.init_type != 'default':
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data,gain=0.000001)
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data,gain=0.000001)
                #m.bias.data.fill_(0.1)

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()


        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(self.num_factors):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class StationaryDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self):
        super(StationaryDecoder, self).__init__()

        print('Using stationary decoder.')

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):

        inputs = inputs.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]

        # Run n prediction steps
        for step in range(0, pred_steps):
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()


class VelocityStepDecoder(nn.Module):
    """MLP decoder module."""

    def __init__(self, delta_T=0.1):
        super(VelocityStepDecoder, self).__init__()
        self.delta_T = delta_T

        print('Using velocity step decoder.')

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):

        # input dimensions ofinputs are [batch, particle, time, state]

        inputs = inputs.transpose(1, 2).contiguous()

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred[:, :, :, 0:2] = last_pred[:, :, :, 0:2] + self.delta_T*last_pred[:, :, :, 2:]
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()
        