"""
This code is based on https://github.com/ethanfetaya/NRI
(MIT licence)
"""

import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

from itertools import permutations, chain
from math import factorial

from os import path

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0) # added dim=0 as implicit choice is deprecated, dim 0 is edgetype due to transpose
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def my_sigmoid(logits, hard=True, sharpness=1.0):

    edges_soft = 1/(1+torch.exp(-sharpness*logits))
    if hard:
        edges_hard = torch.round(edges_soft)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        if edges_soft.is_cuda:
            edges_hard = edges_hard.cuda()
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        edges = Variable(edges_hard - edges_soft.data) + edges_soft
    else:
        edges = edges_soft
    return edges

def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def edge_type_encode(edges): # this is used to gives each 'interaction strength' a unique integer = 0, 1, 2 ..
    unique = np.unique(edges)
    encode = np.zeros(edges.shape)
    for i in range(unique.shape[0]):
        encode += np.where( edges == unique[i], i, 0)
    return encode

def loader_edges_encode(edges, num_atoms): 
    edges = np.reshape(edges, [edges.shape[0], edges.shape[1], num_atoms ** 2])
    edges = np.array(edge_type_encode(edges), dtype=np.int64)
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges = edges[:,:, off_diag_idx]
    return edges

def loader_combine_edges(edges):
    edge_types_list = [ int(np.max(edges[:,i,:]))+1 for i in range(edges.shape[1]) ]
    assert( edge_types_list == sorted(edge_types_list)[::-1] )
    encoded_target = np.zeros( edges[:,0,:].shape )
    base = 1
    for i in reversed(range(edges.shape[1])):
        encoded_target += base*edges[:,i,:]
        base *= edge_types_list[i]
    return encoded_target.astype('int')

def load_data_NRI(batch_size=1, sim_folder='', shuffle=True, data_folder='data'):
    # the edges numpy arrays below are [ num_sims, N, N ]
    loc_train = np.load(path.join(data_folder,sim_folder,'loc_train.npy'))
    vel_train = np.load(path.join(data_folder,sim_folder,'vel_train.npy'))
    edges_train = np.load(path.join(data_folder,sim_folder,'edges_train.npy'))

    loc_valid = np.load(path.join(data_folder,sim_folder,'loc_valid.npy'))
    vel_valid = np.load(path.join(data_folder,sim_folder,'vel_valid.npy'))
    edges_valid = np.load(path.join(data_folder,sim_folder,'edges_valid.npy'))

    loc_test = np.load(path.join(data_folder,sim_folder,'loc_test.npy'))
    vel_test = np.load(path.join(data_folder,sim_folder,'vel_test.npy'))
    edges_test = np.load(path.join(data_folder,sim_folder,'edges_test.npy'))

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)

    edges_train = loader_edges_encode(edges_train, num_atoms)
    edges_valid = loader_edges_encode(edges_valid, num_atoms)
    edges_test = loader_edges_encode(edges_test, num_atoms)

    edges_train = loader_combine_edges(edges_train)
    edges_valid = loader_combine_edges(edges_valid)
    edges_test = loader_combine_edges(edges_test)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min


def load_data_fNRI(batch_size=1, sim_folder='', shuffle=True, data_folder='data'):
    # the edges numpy arrays below are [ num_sims, N, N ]
    loc_train = np.load(path.join(data_folder,sim_folder,'loc_train.npy'))
    vel_train = np.load(path.join(data_folder,sim_folder,'vel_train.npy'))
    edges_train = np.load(path.join(data_folder,sim_folder,'edges_train.npy'))

    loc_valid = np.load(path.join(data_folder,sim_folder,'loc_valid.npy'))
    vel_valid = np.load(path.join(data_folder,sim_folder,'vel_valid.npy'))
    edges_valid = np.load(path.join(data_folder,sim_folder,'edges_valid.npy'))

    loc_test = np.load(path.join(data_folder,sim_folder,'loc_test.npy'))
    vel_test = np.load(path.join(data_folder,sim_folder,'vel_test.npy'))
    edges_test = np.load(path.join(data_folder,sim_folder,'edges_test.npy'))

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)

    edges_train = loader_edges_encode( edges_train, num_atoms )
    edges_valid = loader_edges_encode( edges_valid, num_atoms )
    edges_test = loader_edges_encode( edges_test, num_atoms )

    edges_train = torch.LongTensor(edges_train)
    edges_valid = torch.LongTensor(edges_valid)
    edges_test =  torch.LongTensor(edges_test)

    feat_train = torch.FloatTensor(feat_train)
    feat_valid = torch.FloatTensor(feat_valid)
    feat_test = torch.FloatTensor(feat_test)

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = data_norm + \
           data_norm.transpose(2, 3) - \
           2 * torch.matmul(data, data.transpose(2, 3))
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(np.all(np.vstack((dist > thresholds[i],
                                             dist <= thresholds[i + 1])), 0))[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))  # normalisation here is (batch * num atoms)


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))

def kl_categorical_uniform_var(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return (kl_div.sum(dim=1) / num_atoms).var() 


def nll_gaussian(preds, target, variance, add_const=False):
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1)) # normalisation here is (batch * num atoms)

def nll_gaussian_var(preds, target, variance, add_const=False):
    # returns the variance over the batch of the reconstruction loss
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return (neg_log_p.sum(dim=1)/target.size(1)).var()



def true_flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def KL_between_blocks(prob_list, num_atoms, eps=1e-16):
    # Return a list of the mutual information between every block pair
    KL_list = []
    for i in range(len(prob_list)):
        for j in range(len(prob_list)):
            if i != j:
                KL = prob_list[i] *( torch.log(prob_list[i] + eps) - torch.log(prob_list[j] + eps) )
                KL_list.append( KL.sum() / (num_atoms * prob_list[i].size(0)) )
                KL = prob_list[i] *( torch.log(prob_list[i] + eps) - torch.log( true_flip(prob_list[j],-1) + eps) )
                KL_list.append( KL.sum() / (num_atoms * prob_list[i].size(0)) )  
    return KL_list


def decode_target( target, num_edge_types_list ):
    target_list = []
    base = np.prod(num_edge_types_list)
    for i in range(len(num_edge_types_list)):
        base /= num_edge_types_list[i]
        target_list.append( target//base )
        target = target % base
    return target_list

def encode_target_list( target_list, edge_types_list ):
    encoded_target = np.zeros( target_list[0].shape )
    base = 1
    for i in reversed(range(len(target_list))):
        encoded_target += base*np.array(target_list[i])
        base *= edge_types_list[i]
    return encoded_target.astype('int')

def edge_accuracy_perm_NRI_batch(preds, target, num_edge_types_list):
    # permutation edge accuracy calculator for the standard NRI model
    # return the maximum accuracy of the batch over the permutations of the edge labels
    # also returns a one-hot encoding of the number which represents this permutation
    # also returns the accuracies for the individual factor graphs 

    _, preds = preds.max(-1)    # returns index of max in each z_ij to reduce dim by 1

    num_edge_types = np.prod(num_edge_types_list)
    preds = np.eye(num_edge_types)[np.array(preds.cpu())]  # this is nice way to turn integers into one-hot vectors
    target = np.array(target.cpu())

    perms = [p for p in permutations(range(num_edge_types))] # list of edge type permutations
    # in the below, for each permutation of edge-types, permute preds, then take argmax to go from one-hot to integers
    # then compare to target, compute accuracy
    acc = np.array([np.mean(np.equal(target, np.argmax(preds[:,:,p], axis=-1),dtype=object)) for p in perms])
    max_acc, idx = np.amax(acc), np.argmax(acc)
    preds_deperm = np.argmax(preds[:,:,perms[idx]], axis=-1)

    target_list = decode_target( target, num_edge_types_list )
    preds_deperm_list = decode_target( preds_deperm, num_edge_types_list )

    blocks_acc = [ np.mean(np.equal(target_list[i], preds_deperm_list[i], dtype=object),axis=-1)
                   for i in range(len(target_list)) ]
    acc = np.mean(np.equal(target, preds_deperm ,dtype=object), axis=-1)
    blocks_acc = np.swapaxes(np.array(blocks_acc),0,1)

    idx_onehot = np.eye(len(perms))[np.array(idx)]
    return acc, idx_onehot, blocks_acc

def edge_accuracy_perm_NRI(preds, targets, num_edge_types_list):
    acc_batch, perm_code_onehot, acc_blocks_batch = edge_accuracy_perm_NRI_batch(preds, targets, num_edge_types_list)
    
    acc = np.mean(acc_batch)
    acc_var = np.var(acc_batch)
    acc_blocks = np.mean(acc_blocks_batch, axis=0)
    acc_var_blocks = np.var(acc_blocks_batch, axis=0)

    return acc, perm_code_onehot, acc_blocks, acc_var, acc_var_blocks


def edge_accuracy_perm_fNRI_batch(preds_list, targets, num_edge_types_list):
    # permutation edge accuracy calculator for the fNRI model
    # return the maximum accuracy of the batch over the permutations of the edge labels
    # also returns a one-hot encoding of the number which represents this permutation
    # also returns the accuracies for the individual factor graphs

    target_list = [ targets[:,i,:].cpu() for i in range(targets.shape[1])]
    preds_list = [ pred.max(-1)[1].cpu() for pred in preds_list]
    preds = encode_target_list(preds_list, num_edge_types_list)
    target = encode_target_list(target_list, num_edge_types_list)

    target_list = [ np.array(t.cpu()).astype('int') for t in target_list ]

    num_edge_types = np.prod(num_edge_types_list)
    preds = np.eye(num_edge_types)[preds]     # this is nice way to turn integers into one-hot vectors

    perms = [p for p in permutations(range(num_edge_types))] # list of edge type permutations
    
    # in the below, for each permutation of edge-types, permute preds, then take argmax to go from one-hot to integers
    # then compare to target to compute accuracy
    acc = np.array([np.mean(np.equal(target, np.argmax(preds[:,:,p], axis=-1),dtype=object)) for p in perms])
    max_acc, idx = np.amax(acc), np.argmax(acc)

    preds_deperm = np.argmax(preds[:,:,perms[idx]], axis=-1)
    preds_deperm_list = decode_target( preds_deperm, num_edge_types_list )

    blocks_acc = [ np.mean(np.equal(target_list[i], preds_deperm_list[i], dtype=object),axis=-1) 
                   for i in range(len(target_list)) ]
    acc = np.mean(np.equal(target, preds_deperm ,dtype=object), axis=-1)
    blocks_acc = np.swapaxes(np.array(blocks_acc),0,1)

    idx_onehot = np.array([0])#np.eye(len(perms))[np.array(idx)]

    return acc, idx_onehot, blocks_acc

def edge_accuracy_perm_fNRI_batch_skipfirst(preds_list, targets, num_factors):
    # permutation edge accuracy calculator for the fNRI model when using skip-first argument 
    # and all factor graphs have two edge types
    # return the maximum accuracy of the batch over the permutations of the edge labels
    # also returns a one-hot encoding of the number which represents this permutation
    # also returns the accuracies for the individual factor graphs

    targets = np.swapaxes(np.array(targets.cpu()),1,2)
    preds = torch.cat( [ torch.unsqueeze(pred.max(-1)[1],-1) for pred in preds_list], -1 )
    preds = np.array(preds.cpu())
    perms = [p for p in permutations(range(num_factors))]

    acc = np.array([np.mean(  np.sum(np.equal(targets, preds[:,:,p],dtype=object),axis=-1)==num_factors  ) for p in perms])
    max_acc, idx = np.amax(acc), np.argmax(acc)

    preds_deperm = preds[:,:,perms[idx]]
    blocks_acc = np.mean(np.equal(targets, preds_deperm, dtype=object),axis=1)
    acc = np.mean(  np.sum(np.equal(targets, preds_deperm,dtype=object),axis=-1)==num_factors, axis=-1)

    idx_onehot = np.eye(len(perms))[np.array(idx)]

    return acc, idx_onehot, blocks_acc


def edge_accuracy_perm_fNRI(preds_list, targets, num_edge_types_list, skip_first=False):

    if skip_first and all(e == 2 for e in num_edge_types_list):
        acc_batch, perm_code_onehot, acc_blocks_batch = edge_accuracy_perm_fNRI_batch_skipfirst(preds_list, targets, len(num_edge_types_list))
    else:
        acc_batch, perm_code_onehot, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(preds_list, targets, num_edge_types_list)
    
    acc = np.mean(acc_batch)
    acc_var = np.var(acc_batch)
    acc_blocks = np.mean(acc_blocks_batch, axis=0)
    acc_var_blocks = np.var(acc_blocks_batch, axis=0)

    return acc, perm_code_onehot, acc_blocks, acc_var, acc_var_blocks

def edge_accuracy_perm_sigmoid_batch(preds, targets):
    # permutation edge accuracy calculator for the sigmoid model
    # return the maximum accuracy of the batch over the permutations of the edge labels
    # also returns a one-hot encoding of the number which represents this permutation
    # also returns the accuracies for the individual factor graph_list

    targets = np.swapaxes(np.array(targets.cpu()),1,2)
    preds = np.array(preds.cpu().detach())
    preds = np.rint(preds).astype('int')
    num_factors = targets.shape[-1]
    perms = [p for p in permutations(range(num_factors))] # list of edge type permutations

    # in the below, for each permutation of edge-types, permute preds, then take argmax to go from one-hot to integers
    # then compare to target to compute accuracy
    acc = np.array([np.mean(  np.sum(np.equal(targets, preds[:,:,p],dtype=object),axis=-1)==num_factors  ) for p in perms])
    max_acc, idx = np.amax(acc), np.argmax(acc)

    preds_deperm = preds[:,:,perms[idx]]
    blocks_acc = np.mean(np.equal(targets, preds_deperm, dtype=object),axis=1)
    acc = np.mean( np.sum(np.equal(targets, preds_deperm,dtype=object),axis=-1)==num_factors, axis=-1)

    idx_onehot = np.eye(len(perms))[np.array(idx)]
    return acc, idx_onehot, blocks_acc


def edge_accuracy_perm_sigmoid(preds, targets):
    acc_batch, perm_code_onehot, acc_blocks_batch= edge_accuracy_perm_sigmoid_batch(preds, targets)
    
    acc = np.mean(acc_batch)
    acc_var = np.var(acc_batch)
    acc_blocks = np.mean(acc_blocks_batch, axis=0)
    acc_var_blocks = np.var(acc_blocks_batch, axis=0)

    return acc, perm_code_onehot, acc_blocks, acc_var, acc_var_blocks
