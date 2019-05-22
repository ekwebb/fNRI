"""
This code is based on https://github.com/ethanfetaya/NRI
(MIT licence)
"""
from __future__ import division
from __future__ import print_function

import time
import argparse
import pickle
import os
import datetime
import csv
import math

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from modules import *

parser = argparse.ArgumentParser()
## arguments related to training ##
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--patience', type=int, default=500,
                    help='Early stopping patience')
parser.add_argument('--encoder-dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay value for L2 regularisation in Adam optimiser')

## arguments related to weight and bias initialisation ##
parser.add_argument('--seed', type=int, default=1, 
                    help='Random seed.')
parser.add_argument('--encoder-init-type',type=str, default='xavier_normal',
                    help='The type of weight initialization to use in the encoder')
parser.add_argument('--encoder-bias-scale',type=float, default=0.1,
                    help='The type of weight initialization to use in the encoder')

## arguments related to changing the model ##
parser.add_argument('--NRI', action='store_true', default=False,
                    help='Use the NRI model, rather than the fNRI model')
parser.add_argument('--edge-types-list', nargs='+', default=[2,2],
                    help='The number of edge types to infer.') # takes arguments from cmd line as: --edge-types-list 2 2
parser.add_argument('--split-point', type=int, default=0,
                    help='The point at which factor graphs are split up in the encoder' )
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')

parser.add_argument('--sigmoid', action='store_true', default=False,
                    help='Use the sfNRI model, rather than the fNRI model')
parser.add_argument('--num-factors', type=int, default=2,
                    help='The number of factor graphs (this is only for sigmoid)')
parser.add_argument('--sigmoid-sharpness', type=float, default=1.,
                    help='Coefficient in the power of the sigmoid function')

## arguments related to the simulation data ##
parser.add_argument('--sim-folder', type=str, default='springcharge_5',
                    help='Name of the folder in the data folder to load simulation data from')
parser.add_argument('--data-folder', type=str, default='data',
                    help='Name of the data folder to load data from')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--dims', type=int, default=4,
                    help='The number of input dimensions (position + velocity).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')

## Saving, loading etc. ##
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--test', action='store_true', default=False,
                    help='Skip training and validation')
parser.add_argument('--no-edge-acc', action='store_true', default=False,
                    help='Skip training and plot accuracy distributions')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
#args.factor = not args.no_factor
args.edge_types_list = list(map(int, args.edge_types_list))
args.edge_types_list.sort(reverse=True)

if all( (isinstance(k, int) and k >= 1) for k in args.edge_types_list):
    if args.NRI:
        edge_types = np.prod(args.edge_types_list)
    else:
        edge_types = sum(args.edge_types_list)
else:
    raise ValueError('Could not compute the edge-types-list')

if args.NRI:
    print('Using NRI model')
    if args.split_point != 0:
        args.split_point = 0
print(args)

if args.prior:
    prior = [ [0.9, 0.1] , [0.9, 0.1] ]  # TODO: hard coded for now
    if not all( prior[i].size == edge_types_list[i] for i in range(len(args.edge_types_list))):
        raise ValueError('Prior is incompatable with the edge types list')
    print("Using prior: "+str(prior))
    log_prior = []
    for i in range(len(args.edge_types_list)):
        prior_i = np.array(prior[i])
        log_prior_i = torch.FloatTensor(np.log(prior))
        log_prior_i = torch.unsqueeze(log_prior_i, 0)
        log_prior_i = torch.unsqueeze(log_prior_i, 0)
        log_prior_i = Variable(log_prior_i)
        log_prior.append(log_prior_i)
    if args.cuda:
        log_prior = log_prior.cuda()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat().replace(':','-')[:-7]
    save_folder = os.path.join(args.save_folder,'exp'+timestamp)
    os.makedirs(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log_csv_file = os.path.join(save_folder, 'log_csv.csv')
    log = open(log_file, 'w')
    log_csv = open(log_csv_file, 'w')
    csv_writer = csv.writer(log_csv, delimiter=',')

    pickle.dump({'args': args}, open(meta_file, "wb"))
    par_file = open(os.path.join(save_folder,'args.txt'),'w')
    print(args,file=par_file)
    par_file.flush
    par_file.close()

else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")


if args.NRI:
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data_NRI(
                                                        args.batch_size, args.sim_folder, shuffle=True, 
                                                        data_folder=args.data_folder)
else:
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data_fNRI(
                                                        args.batch_size, args.sim_folder, shuffle=True,
                                                        data_folder=args.data_folder)


# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms) 
rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

if args.NRI:
    edge_types_list = [ edge_types ]
else:
    edge_types_list = args.edge_types_list

if args.encoder == 'mlp':
    if args.sigmoid:
        encoder = MLPEncoder_sigmoid(args.timesteps * args.dims, args.encoder_hidden,
                                       args.num_factors,args.encoder_dropout,
                                       split_point=args.split_point)
    else:
        encoder = MLPEncoder_multi(args.timesteps * args.dims, args.encoder_hidden,
                                   edge_types_list, args.encoder_dropout,
                                   split_point=args.split_point,
                                   init_type=args.encoder_init_type,
                                   bias_init=args.encoder_bias_scale)

elif args.encoder == 'cnn':
    encoder = CNNEncoder_multi(args.dims, args.encoder_hidden,
                               edge_types_list,
                               args.encoder_dropout,
                               split_point=args.split_point,
                               init_type=args.encoder_init_type)



if args.load_folder:
    print('Loading model from: '+args.load_folder)
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    if not args.cuda:
        encoder.load_state_dict(torch.load(encoder_file,map_location='cpu'))
    else:
        encoder.load_state_dict(torch.load(encoder_file))
    args.save_folder = False

optimizer = optim.Adam(list(encoder.parameters()),
                       lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)


if args.cuda:
    encoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def train(epoch, best_val_loss):
    t = time.time()

    kl_train = []
    kl_list_train = []
    kl_var_list_train = []

    acc_train = []
    acc_blocks_train = []
    acc_var_train = []
    acc_var_blocks_train = []
    
    KLb_train = []
    KLb_blocks_train = []

    ce_train = []

    encoder.train()
    scheduler.step()

    for batch_idx, (data, relations) in enumerate(train_loader): # relations are the ground truth interactions graphs
        if args.cuda:
            data, relations = data.cuda(), relations.cuda()
        data, relations = Variable(data), Variable(relations)

        data_encoder = data[:, :, :args.timesteps, :].contiguous()

        optimizer.zero_grad()

        logits = encoder(data_encoder, rel_rec, rel_send)

        if args.NRI:
            prob = my_softmax(logits, -1) 

            loss_kl = kl_categorical_uniform(prob, args.num_atoms, edge_types)
            loss_kl_split = [ loss_kl ]
            loss_kl_var_split = [ kl_categorical_uniform_var(prob, args.num_atoms, edge_types) ]
            kl_train.append(loss_kl.data.item())
            kl_list_train.append([kl.data.item() for kl in loss_kl_split])
            kl_var_list_train.append([kl_var.data.item() for kl_var in loss_kl_var_split])
            
            KLb_train.append( 0 )
            KLb_blocks_train.append([0])

            preds = np.array(logits.max(-1)[1].cpu())
            targets = np.array(relations.cpu())
            preds_list = decode_target( preds, args.edge_types_list )
            target_list = decode_target( targets, args.edge_types_list )

            acc = np.mean(np.equal(targets, preds,dtype=object))
            acc_blocks = np.array([ np.mean(np.equal(target_list[i], preds_list[i], dtype=object))
                                    for i in range(len(target_list)) ])
            acc_var = np.var(np.mean(np.equal(targets, preds,dtype=object),axis=-1))
            acc_var_blocks = np.array([ np.var(np.mean(np.equal(target_list[i], preds_list[i], dtype=object),axis=-1))
                                        for i in range(len(target_list)) ])

            logits = logits.view(-1, edge_types)
            relations = relations.view(-1)

            loss = F.cross_entropy(logits, relations)

        elif args.sigmoid:
            edges = 1/(1+torch.exp(-args.sigmoid_sharpness*logits))
            
            targets = np.swapaxes(np.array(relations.cpu()),1,2)
            preds = np.array(edges.cpu().detach())
            preds = np.rint(preds).astype('int')

            acc = np.mean(  np.sum(np.equal(targets, preds,dtype=object),axis=-1)==args.num_factors )
            acc_blocks = np.mean(np.equal(targets, preds, dtype=object),axis=(0,1))
            acc_var = np.var(np.mean(  np.sum(np.equal(targets, preds,dtype=object), axis=-1)==args.num_factors, axis=1))
            acc_var_blocks = np.var(np.mean(np.equal(targets, preds, dtype=object), axis=1), axis=0)

            edges = edges.view(-1)
            relations = relations.transpose(1,2).type(torch.FloatTensor).contiguous().view(-1)
            if args.cuda:
                relations = relations.cuda()
            loss = F.binary_cross_entropy( edges, relations )

            kl_train.append(0)
            kl_list_train.append([0])
            kl_var_list_train.append([0])
            KLb_train.append( 0 )
            KLb_blocks_train.append( [0] )

        else:
            # dim of logits, edges and prob are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
            logits_split = torch.split(logits, args.edge_types_list, dim=-1)

            prob_split = [my_softmax(logits_i, -1) for logits_i in logits_split ] 

            if args.prior:
                loss_kl_split = [kl_categorical(prob_split[type_idx], log_prior[type_idx], args.num_atoms) 
                                 for type_idx in range(len(args.edge_types_list)) ]
                loss_kl = sum(loss_kl_split)  
            else:
                loss_kl_split = [ kl_categorical_uniform(prob_split[type_idx], args.num_atoms, 
                                                         args.edge_types_list[type_idx]) 
                                  for type_idx in range(len(args.edge_types_list)) ]
                loss_kl = sum(loss_kl_split)

                loss_kl_var_split = [ kl_categorical_uniform_var(prob_split[type_idx], args.num_atoms, 
                                                                 args.edge_types_list[type_idx]) 
                                      for type_idx in range(len(args.edge_types_list)) ]

            kl_train.append(loss_kl.data.item())
            kl_list_train.append([kl.data.item() for kl in loss_kl_split])
            kl_var_list_train.append([kl_var.data.item() for kl_var in loss_kl_var_split])
            KLb_blocks = KL_between_blocks(prob_split, args.num_atoms)
            KLb_train.append(sum(KLb_blocks).data.item())
            KLb_blocks_train.append([KL.data.item() for KL in KLb_blocks])

            targets = np.swapaxes(np.array(relations.cpu()),1,2)
            preds = torch.cat( [ torch.unsqueeze(pred.max(-1)[1],-1) for pred in logits_split], -1 )
            preds = np.array(preds.cpu())

            acc = np.mean(  np.sum(np.equal(targets, preds,dtype=object),axis=-1)==len(args.edge_types_list)  )                
            acc_blocks = np.mean(np.equal(targets, preds, dtype=object),axis=(0,1))
            acc_var = np.var(np.mean(np.sum(np.equal(targets, preds,dtype=object),axis=-1)==len(args.edge_types_list), axis=-1))
            acc_var_blocks = np.var(np.mean(np.equal(targets, preds, dtype=object), axis=1),axis=0)

            loss = 0
            for i in range(len(args.edge_types_list)):
                logits_i = logits_split[i].view(-1, args.edge_types_list[i])
                relations_i = relations[:,i,:].contiguous().view(-1)
                loss += F.cross_entropy(logits_i, relations_i)                                               


        loss.backward()
        optimizer.step()

        acc_train.append(acc)
        acc_blocks_train.append(acc_blocks)
        acc_var_train.append(acc_var)
        acc_var_blocks_train.append(acc_var_blocks)

        ce_train.append(loss.data.item())



    kl_val = []
    kl_list_val = []
    kl_var_list_val = []
    
    acc_val = []
    acc_blocks_val = []
    acc_var_val = []
    acc_var_blocks_val = []

    KLb_val = []
    KLb_blocks_val = [] # KL between blocks list

    ce_val = []

    encoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        with torch.no_grad():
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()

            data_encoder = data[:, :, :args.timesteps, :].contiguous()

            logits = encoder(data_encoder, rel_rec, rel_send)

            if args.NRI:
                prob = my_softmax(logits, -1)

                loss_kl = kl_categorical_uniform(prob, args.num_atoms, edge_types)
                loss_kl_split = [ loss_kl ]
                loss_kl_var_split = [ kl_categorical_uniform_var(prob, args.num_atoms, edge_types) ]
                kl_val.append(loss_kl.data.item())
                kl_list_val.append([kl.data.item() for kl in loss_kl_split])
                kl_var_list_val.append([kl_var.data.item() for kl_var in loss_kl_var_split])
                
                KLb_val.append( 0 )
                KLb_blocks_val.append([0])

                preds = np.array(logits.max(-1)[1].cpu())
                targets = np.array(relations.cpu())
                preds_list = decode_target( preds, args.edge_types_list )
                target_list = decode_target( targets, args.edge_types_list )

                acc = np.mean(np.equal(targets, preds,dtype=object))
                acc_blocks = np.array([ np.mean(np.equal(target_list[i], preds_list[i], dtype=object))
                                        for i in range(len(target_list)) ])
                acc_var = np.var(np.mean(np.equal(targets, preds,dtype=object),axis=-1))
                acc_var_blocks = np.array([ np.var(np.mean(np.equal(target_list[i], preds_list[i], dtype=object),axis=-1))
                                            for i in range(len(target_list)) ])

                logits = logits.view(-1, edge_types)
                relations = relations.view(-1)

                loss = F.cross_entropy(logits, relations)

            elif args.sigmoid:
                edges = 1/(1+torch.exp(-args.sigmoid_sharpness*logits))
                
                targets = np.swapaxes(np.array(relations.cpu()),1,2)
                preds = np.array(edges.cpu().detach())
                preds = np.rint(preds).astype('int')

                acc = np.mean(  np.sum(np.equal(targets, preds,dtype=object),axis=-1)==args.num_factors )
                acc_blocks = np.mean(np.equal(targets, preds, dtype=object),axis=(0,1))
                acc_var = np.var(np.mean(  np.sum(np.equal(targets, preds,dtype=object), axis=-1)==args.num_factors, axis=1))
                acc_var_blocks = np.var(np.mean(np.equal(targets, preds, dtype=object), axis=1), axis=0)

                edges = edges.view(-1)
                relations = relations.transpose(1,2).type(torch.FloatTensor).contiguous().view(-1)
                if args.cuda:
                    relations = relations.cuda()
                loss = F.binary_cross_entropy( edges, relations )

                kl_val.append(0)
                kl_list_val.append([0])
                kl_var_list_val.append([0])
                KLb_val.append( 0 )
                KLb_blocks_val.append( [0] )

            else:
                logits_split = torch.split(logits, args.edge_types_list, dim=-1)
                prob_split = [my_softmax(logits_i, -1) for logits_i in logits_split ] 

                if args.prior:
                    loss_kl_split = [kl_categorical(prob_split[type_idx], log_prior[type_idx], args.num_atoms) 
                                     for type_idx in range(len(args.edge_types_list)) ]
                    loss_kl = sum(loss_kl_split)  
                else:
                    loss_kl_split = [ kl_categorical_uniform(prob_split[type_idx], args.num_atoms, 
                                                             args.edge_types_list[type_idx]) 
                                      for type_idx in range(len(args.edge_types_list)) ]
                    loss_kl = sum(loss_kl_split)

                    loss_kl_var_split = [ kl_categorical_uniform_var(prob_split[type_idx], args.num_atoms, 
                                                                     args.edge_types_list[type_idx]) 
                                          for type_idx in range(len(args.edge_types_list)) ]

                kl_val.append(loss_kl.data.item())
                kl_list_val.append([kl.data.item() for kl in loss_kl_split])
                kl_var_list_val.append([kl_var.data.item() for kl_var in loss_kl_var_split])

                targets = np.swapaxes(np.array(relations.cpu()),1,2)
                preds = torch.cat( [ torch.unsqueeze(pred.max(-1)[1],-1) for pred in logits_split], -1 )
                preds = np.array(preds.cpu())

                acc = np.mean(  np.sum(np.equal(targets, preds,dtype=object),axis=-1)==len(args.edge_types_list)  )                
                acc_blocks = np.mean(np.equal(targets, preds, dtype=object),axis=(0,1))
                acc_var = np.var(np.mean(np.sum(np.equal(targets, preds,dtype=object),axis=-1)==len(args.edge_types_list), axis=-1))
                acc_var_blocks = np.var(np.mean(np.equal(targets, preds, dtype=object), axis=1),axis=0)

                loss = 0
                for i in range(len(args.edge_types_list)):
                    logits_i = logits_split[i].view(-1, args.edge_types_list[i])
                    relations_i = relations[:,i,:].contiguous().view(-1)
                    loss += F.cross_entropy(logits_i, relations_i)                                               

                KLb_blocks = KL_between_blocks(prob_split, args.num_atoms)
                KLb_val.append(sum(KLb_blocks).data.item())
                KLb_blocks_val.append([KL.data.item() for KL in KLb_blocks])


            acc_val.append(acc)
            acc_blocks_val.append(acc_blocks)
            acc_var_val.append(acc_var)
            acc_var_blocks_val.append(acc_var_blocks)

            ce_val.append(loss.data.item())


    print('Epoch: {:03d}'.format(epoch),
          'time: {:.1f}s'.format(time.time() - t))
    print('ce_trn: {:.5f}'.format(np.mean(ce_train)),
          'kl_trn: {:.5f}'.format(np.mean(kl_train)),
          'acc_trn: {:.5f}'.format(np.mean(acc_train)),
          'KLb_trn: {:.5f}'.format(np.mean(KLb_train)),
          'acc_b_trn: '+str( np.around(np.mean(np.array(acc_blocks_train),axis=0),4 ) ),
          'kl_trn: '+str( np.around(np.mean(np.array(kl_list_train),axis=0),4 ) )
          ) 
    print('ce_val: {:.5f}'.format(np.mean(ce_val)),
          'kl_val: {:.5f}'.format(np.mean(kl_val)),
          'acc_val: {:.5f}'.format(np.mean(acc_val)),
          'KLb_val: {:.5f}'.format(np.mean(KLb_val)),
          'acc_b_val: '+str( np.around(np.mean(np.array(acc_blocks_val),axis=0),4 ) ),
          'kl_val: '+str( np.around(np.mean(np.array(kl_list_val),axis=0),4 ) ),
          ) 
    print('Epoch: {:04d}'.format(epoch),
          'time: {:.4f}s'.format(time.time() - t),
           file=log)
    print('ce_trn: {:.5f}'.format(np.mean(ce_train)),
          'kl_trn: {:.5f}'.format(np.mean(kl_train)),
          'acc_trn: {:.5f}'.format(np.mean(acc_train)),
          'KLb_trn: {:.5f}'.format(np.mean(KLb_train)),
          'acc_b_trn: '+str( np.around(np.mean(np.array(acc_blocks_train),axis=0),4 ) ),
          'kl_trn: '+str( np.around(np.mean(np.array(kl_list_train),axis=0),4 ) ),
           file=log )
    print('ce_val: {:.5f}'.format(np.mean(ce_val)),
          'kl_val: {:.5f}'.format(np.mean(kl_val)),
          'acc_val: {:.5f}'.format(np.mean(acc_val)),
          'KLb_val: {:.5f}'.format(np.mean(KLb_val)),
          'acc_b_val: '+str( np.around(np.mean(np.array(acc_blocks_val),axis=0),4 ) ),
          'kl_val: '+str( np.around(np.mean(np.array(kl_list_val),axis=0),4 ) ),
          file=log)
    if epoch == 0:
        labels =  [ 'epoch', 'ce trn', 'kl trn', 'KLb trn', 'acc trn' ]
        labels += [ 'b'+str(i)+ ' acc trn' for i in range( len(args.edge_types_list) ) ]
        labels += [ 'b'+str(i)+ ' kl trn' for i in range( len(kl_list_train[0]) ) ]
        labels += [ 'b'+str(i)+' kl var trn' for i in range( len(kl_list_train[0]) ) ]
        labels += [ 'acc var trn']  + [ 'b'+str(i)+' acc var trn' for i in range( len(args.edge_types_list) ) ]
        labels += [ 'ce val', 'kl val', 'KLb val', 'acc val' ]
        labels += [ 'b'+str(i)+ ' acc val' for i in range( len(args.edge_types_list) ) ] 
        labels += [ 'b'+str(i)+ ' kl val' for i in range( len(kl_list_val[0]) ) ]
        labels += [ 'b'+str(i)+' kl var val' for i in range( len(kl_list_val[0]) ) ]
        labels += [ 'acc var val']  + [ 'b'+str(i)+' acc var val' for i in range( len(args.edge_types_list) ) ]
        csv_writer.writerow( labels )


    csv_writer.writerow( [epoch, np.mean(ce_train), np.mean(kl_train), np.mean(KLb_train), np.mean(acc_train)] +
                         list(np.mean(np.array(acc_blocks_train),axis=0)) + 
                         list(np.mean(np.array(kl_list_train),axis=0)) +
                         list(np.mean(np.array(kl_var_list_train),axis=0)) +
                         [np.mean(acc_var_train)] + list(np.mean(np.array(acc_var_blocks_train),axis=0)) +
                         [ np.mean(ce_val), np.mean(kl_val), np.mean(KLb_val), np.mean(acc_val) ] +
                         list(np.mean(np.array(acc_blocks_val  ),axis=0)) +
                         list(np.mean(np.array(kl_list_val),axis=0)) +
                         list(np.mean(np.array(kl_var_list_val),axis=0)) +
                         [np.mean(acc_var_val)] + list(np.mean(np.array(acc_var_blocks_val),axis=0))
                        )

    log.flush()
    if args.save_folder and np.mean(acc_val) > best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        print('Best model so far, saving...')
    return np.mean(acc_val)


def test():
    t = time.time()

    ce_test = []

    kl_test = []
    kl_list_test = []
    kl_var_list_test = []
    
    acc_test = []
    acc_blocks_test = []
    acc_var_test = []
    acc_var_blocks_test = []

    KLb_test = []
    KLb_blocks_test = [] # KL between blocks list

    encoder.eval()
    if not args.cuda:
        encoder.load_state_dict(torch.load(encoder_file,map_location='cpu'))
    else:
        encoder.load_state_dict(torch.load(encoder_file))

    for batch_idx, (data, relations) in enumerate(test_loader):
        with torch.no_grad():
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()

            data_encoder = data[:, :, :args.timesteps, :].contiguous()

            logits = encoder(data_encoder, rel_rec, rel_send)

            if args.NRI:
                prob = my_softmax(logits, -1) 

                loss_kl = kl_categorical_uniform(prob, args.num_atoms, edge_types)
                loss_kl_split = [ loss_kl ]
                loss_kl_var_split = [ kl_categorical_uniform_var(prob, args.num_atoms, edge_types) ]
                kl_test.append(loss_kl.data.item())
                kl_list_test.append([kl.data.item() for kl in loss_kl_split])
                kl_var_list_test.append([kl_var.data.item() for kl_var in loss_kl_var_split])
                
                KLb_test.append( 0 )
                KLb_blocks_test.append([0])

                preds = np.array(logits.max(-1)[1].cpu())
                targets = np.array(relations.cpu())
                preds_list = decode_target( preds, args.edge_types_list )
                target_list = decode_target( targets, args.edge_types_list )

                acc = np.mean(np.equal(targets, preds,dtype=object))
                acc_blocks = np.array([ np.mean(np.equal(target_list[i], preds_list[i], dtype=object))
                                        for i in range(len(target_list)) ])
                acc_var = np.var(np.mean(np.equal(targets, preds,dtype=object),axis=-1))
                acc_var_blocks = np.array([ np.var(np.mean(np.equal(target_list[i], preds_list[i], dtype=object),axis=-1))
                                            for i in range(len(target_list)) ])

                logits = logits.view(-1, edge_types)
                relations = relations.view(-1)

                loss = F.cross_entropy(logits, relations)

            elif args.sigmoid:
                edges = 1/(1+torch.exp(-args.sigmoid_sharpness*logits))
                
                targets = np.swapaxes(np.array(relations.cpu()),1,2)
                preds = np.array(edges.cpu().detach())
                preds = np.rint(preds).astype('int')

                acc = np.mean(  np.sum(np.equal(targets, preds,dtype=object),axis=-1)==args.num_factors )
                acc_blocks = np.mean(np.equal(targets, preds, dtype=object),axis=(0,1))
                acc_var = np.var(np.mean(  np.sum(np.equal(targets, preds,dtype=object), axis=-1)==args.num_factors, axis=1))
                acc_var_blocks = np.var(np.mean(np.equal(targets, preds, dtype=object), axis=1), axis=0)

                edges = edges.view(-1)
                relations = relations.transpose(1,2).type(torch.FloatTensor).contiguous().view(-1)
                if args.cuda:
                    relations = relations.cuda()
                loss = F.binary_cross_entropy( edges, relations )

                kl_test.append(0)
                kl_list_test.append([0])
                kl_var_list_test.append([0])
                KLb_test.append( 0 )
                KLb_blocks_test.append( [0] )

            else:
                logits_split = torch.split(logits, args.edge_types_list, dim=-1)

                prob_split = [my_softmax(logits_i, -1) for logits_i in logits_split ] 

                if args.prior:
                    loss_kl_split = [kl_categorical(prob_split[type_idx], log_prior[type_idx], args.num_atoms) 
                                     for type_idx in range(len(args.edge_types_list)) ]
                    loss_kl = sum(loss_kl_split)  
                else:
                    loss_kl_split = [ kl_categorical_uniform(prob_split[type_idx], args.num_atoms, 
                                                             args.edge_types_list[type_idx]) 
                                      for type_idx in range(len(args.edge_types_list)) ]
                    loss_kl = sum(loss_kl_split)

                    loss_kl_var_split = [ kl_categorical_uniform_var(prob_split[type_idx], args.num_atoms, 
                                                                     args.edge_types_list[type_idx]) 
                                          for type_idx in range(len(args.edge_types_list)) ]

                kl_test.append(loss_kl.data.item())
                kl_list_test.append([kl.data.item() for kl in loss_kl_split])
                kl_var_list_test.append([kl_var.data.item() for kl_var in loss_kl_var_split])

                targets = np.swapaxes(np.array(relations.cpu()),1,2)
                preds = torch.cat( [ torch.unsqueeze(pred.max(-1)[1],-1) for pred in logits_split], -1 )
                preds = np.array(preds.cpu())

                acc = np.mean(  np.sum(np.equal(targets, preds,dtype=object),axis=-1)==len(args.edge_types_list)  )                
                acc_blocks = np.mean(np.equal(targets, preds, dtype=object),axis=(0,1))
                acc_var = np.var(np.mean(np.sum(np.equal(targets, preds,dtype=object),axis=-1)==len(args.edge_types_list), axis=-1))
                acc_var_blocks = np.var(np.mean(np.equal(targets, preds, dtype=object), axis=1),axis=0)

                loss = 0
                for i in range(len(args.edge_types_list)):
                    logits_i = logits_split[i].view(-1, args.edge_types_list[i])
                    relations_i = relations[:,i,:].contiguous().view(-1)
                    loss += F.cross_entropy(logits_i, relations_i)                                               

                KLb_blocks = KL_between_blocks(prob_split, args.num_atoms)
                KLb_test.append(sum(KLb_blocks).data.item())
                KLb_blocks_test.append([KL.data.item() for KL in KLb_blocks])    

            ce_test.append(loss.data.item())
            acc_test.append(acc)
            acc_blocks_test.append(acc_blocks)
            acc_var_test.append(acc_var)
            acc_var_blocks_test.append(acc_var_blocks)


    print('--------------------------------')
    print('------------Testing-------------')
    print('--------------------------------')
    print('ce_test: {:.2f}'.format(np.mean(ce_test)),
          'kl_test: {:.5f}'.format(np.mean(kl_test)),
          'acc_test: {:.5f}'.format(np.mean(acc_test)),
          'acc_var_test: {:.5f}'.format(np.mean(acc_var_test)),
          'KLb_test: {:.5f}'.format(np.mean(KLb_test)),
          'time: {:.1f}s'.format(time.time() - t))
    print('acc_b_test: '+str( np.around(np.mean(np.array(acc_blocks_test),axis=0),4 ) ),
          'acc_var_b_test: '+str( np.around(np.mean(np.array(acc_var_blocks_test),axis=0),4 ) ),
          'kl_test: '+str( np.around(np.mean(np.array(kl_list_test),axis=0),4 ) )) 
    if args.save_folder:
        print('--------------------------------', file=log)
        print('------------Testing-------------', file=log)
        print('--------------------------------', file=log)
        print('ce_test: {:.2f}'.format(np.mean(ce_test)),
              'kl_test: {:.5f}'.format(np.mean(kl_test)),
              'acc_test: {:.5f}'.format(np.mean(acc_test)),
              'acc_var_test: {:.5f}'.format(np.mean(acc_var_test)),
              'KLb_test: {:.5f}'.format(np.mean(KLb_test)),
              'time: {:.1f}s'.format(time.time() - t),
              file=log)
        print('acc_b_test: '+str( np.around(np.mean(np.array(acc_blocks_test),axis=0),4 ) ),
              'acc_var_b_test: '+str( np.around(np.mean(np.array(acc_var_blocks_test),axis=0),4 ) ),
              'kl_test: '+str( np.around(np.mean(np.array(kl_list_test),axis=0),4 ) ),
              file=log) 
        log.flush()


# Train model
if not args.test:
    t_total = time.time()
    best_val_loss = 0
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss = train(epoch, best_val_loss)
        if val_loss > best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        if epoch - best_epoch > args.patience and epoch > 99:
            break
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

print('Reloading best model')
test()
if log is not None:
    print(save_folder)
    log.close()
    log_csv.close()
