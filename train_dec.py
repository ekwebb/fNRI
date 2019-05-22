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
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--patience', type=int, default=500,
                    help='Early stopping patience')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dont-split-data', action='store_true', default=False,
                    help='Whether to not split training and validation data into two parts')
parser.add_argument('--split-enc-only', action='store_true', default=False,
                    help='Whether to give the encoder the first half of trajectories \
                          and the decoder the whole of the trajectories')

## arguments related to loss function ##
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')

## arguments related to weight and bias initialisation ##
parser.add_argument('--seed', type=int, default=1, 
                    help='Random seed.')
parser.add_argument('--decoder-init-type',type=str, default='default',
                    help='The type of weight initialization to use in the decoder')


## arguments related to changing the model ##
parser.add_argument('--NRI', action='store_true', default=False,
                    help='Use the NRI model, rather than the fNRI model')
parser.add_argument('--sigmoid', action='store_true', default=False,
                    help='Use the sfNRI model, rather than the fNRI model')
parser.add_argument('--edge-types-list', nargs='+', default=[2,2],
                    help='The number of edge types to infer.') # takes arguments from cmd line as: --edge-types-list 2 2
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip the first edge type in each block in the decoder, i.e. it represents no-edge.')
parser.add_argument('--full-graph', action='store_true', default=False,
                    help='Use a fixed fully connected graph rather than the ground truth labels')
parser.add_argument('--num-factors', type=int, default=2,
                    help='The number of factor graphs (this is only for sfNRI model, replaces edge-types-list)')

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
parser.add_argument('--plot', action='store_true', default=False,
                    help='Skip training and plot trajectories against actual')


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

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat().replace(':','-')[:-7]
    save_folder = os.path.join(args.save_folder,'exp'+timestamp)
    os.makedirs(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

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

if args.decoder == 'mlp':
    if args.sigmoid:
        decoder = MLPDecoder_sigmoid(n_in_node=args.dims,
                                     num_factors=args.num_factors,
                                     msg_hid=args.decoder_hidden,
                                     msg_out=args.decoder_hidden,
                                     n_hid=args.decoder_hidden,
                                     do_prob=args.decoder_dropout,
                                     init_type=args.decoder_init_type)
    else:
        decoder = MLPDecoder_multi(n_in_node=args.dims,
                                 edge_types=edge_types,
                                 edge_types_list=edge_types_list,
                                 msg_hid=args.decoder_hidden,
                                 msg_out=args.decoder_hidden,
                                 n_hid=args.decoder_hidden,
                                 do_prob=args.decoder_dropout,
                                 skip_first=args.skip_first,
                                 init_type=args.decoder_init_type)

elif args.decoder == 'stationary':
    decoder = StationaryDecoder()

elif args.decoder == 'velocity':
    decoder = VelocityStepDecoder()

if args.load_folder:
    print('Loading model from: '+args.load_folder)
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    if not args.cuda:
        decoder.load_state_dict(torch.load(decoder_file,map_location='cpu'))
    else:
        decoder.load_state_dict(torch.load(decoder_file))
    args.save_folder = False

optimizer = optim.Adam(list(decoder.parameters()), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)


if args.cuda:
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)


def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    nll_var_train = []
    mse_train = []

    decoder.train()
    scheduler.step()
    if not args.plot:
        for batch_idx, (data, relations) in enumerate(train_loader): # relations are the ground truth interactions graphs

            optimizer.zero_grad()

            if args.full_graph:
                zeros = torch.zeros([data.size(0), rel_rec.size(0)])
                ones = torch.ones([data.size(0), rel_rec.size(0)])
                if args.NRI:
                    stack = [ ones ] + [ zeros for _ in range(edge_types-1) ]
                    rel_type_onehot = torch.stack(stack, -1)
                elif args.sigmoid:
                    stack = [ ones for _ in range(args.num_factors) ]
                    rel_type_onehot = torch.stack(stack, -1)
                else:
                    stack = []
                    for i in range(len(args.edge_types_list)):
                        stack += [ ones ] + [ zeros for _ in range(args.edge_types_list[i]-1) ]
                    rel_type_onehot = torch.stack(stack, -1)

            else:
                if args.NRI:
                    rel_type_onehot = torch.FloatTensor(data.size(0), rel_rec.size(0), edge_types)
                    rel_type_onehot.zero_()
                    rel_type_onehot.scatter_(2, relations.view(data.size(0), -1, 1), 1)
                elif args.sigmoid:
                    rel_type_onehot = relations.transpose(1,2).type(torch.FloatTensor)
                else:
                    rel_type_onehot = [ torch.FloatTensor(data.size(0), rel_rec.size(0), types) for types in args.edge_types_list ]
                    rel_type_onehot = [ rel.zero_() for rel in rel_type_onehot ]
                    rel_type_onehot = [ rel_type_onehot[i].scatter_(2, relations[:,i,:].view(data.size(0), -1, 1), 1) for i in range(len(rel_type_onehot)) ]
                    rel_type_onehot = torch.cat( rel_type_onehot, dim=-1 )

            if args.dont_split_data:
                data_decoder = data[:, :, :args.timesteps, :]
            elif args.split_enc_only:
                data_decoder = data
            else:
                assert (data.size(2) - args.timesteps) >= args.timesteps
                data_decoder = data[:, :, -args.timesteps:, :]

            if args.cuda:
                data_decoder, rel_type_onehot = data_decoder.cuda(), rel_type_onehot.cuda()
            data_decoder = data_decoder.contiguous()

            data_decoder, rel_type_onehot = Variable(data_decoder), Variable(rel_type_onehot)
            
            target = data_decoder[:, :, 1:, :] # dimensions are [batch, particle, time, state]
            output = decoder(data_decoder, rel_type_onehot, rel_rec, rel_send, args.prediction_steps)

            loss_nll = nll_gaussian(output, target, args.var)
            loss_nll_var = nll_gaussian_var(output, target, args.var)


            loss_nll.backward() 
            optimizer.step()

            mse_train.append(F.mse_loss(output, target).data.item()) 
            nll_train.append(loss_nll.data.item())
            nll_var_train.append(loss_nll_var.data.item())

            
    nll_val = []
    nll_var_val = []
    mse_val = []

    nll_M_val = []
    nll_M_var_val = []

    decoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        with torch.no_grad():

            if args.full_graph:
                zeros = torch.zeros([data.size(0), rel_rec.size(0)])
                ones = torch.ones([data.size(0), rel_rec.size(0)])
                if args.NRI:
                    stack = [ ones ] + [ zeros for _ in range(edge_types-1) ]
                    rel_type_onehot = torch.stack(stack, -1)
                elif args.sigmoid:
                    stack = [ ones for _ in range(args.num_factors) ]
                    rel_type_onehot = torch.stack(stack, -1)
                else:
                    stack = []
                    for i in range(len(args.edge_types_list)):
                        stack += [ ones ] + [ zeros for _ in range(args.edge_types_list[i]-1) ]
                    rel_type_onehot = torch.stack(stack, -1)

            else:
                if args.NRI:
                    rel_type_onehot = torch.FloatTensor(data.size(0), rel_rec.size(0), edge_types)
                    rel_type_onehot.zero_()
                    rel_type_onehot.scatter_(2, relations.view(data.size(0), -1, 1), 1)
                elif args.sigmoid:
                    rel_type_onehot = relations.transpose(1,2).type(torch.FloatTensor)
                else:
                    rel_type_onehot = [ torch.FloatTensor(data.size(0), rel_rec.size(0), types) for types in args.edge_types_list ]
                    rel_type_onehot = [ rel.zero_() for rel in rel_type_onehot ]
                    rel_type_onehot = [ rel_type_onehot[i].scatter_(2, relations[:,i,:].view(data.size(0), -1, 1), 1) for i in range(len(rel_type_onehot)) ]
                    rel_type_onehot = torch.cat( rel_type_onehot, dim=-1 )

            if args.dont_split_data:
                data_decoder = data[:, :, :args.timesteps, :]
            elif args.split_enc_only:
                data_decoder = data
            else:
                assert (data.size(2) - args.timesteps) >= args.timesteps
                data_decoder = data[:, :, -args.timesteps:, :]

            if args.cuda:
                data_decoder, rel_type_onehot = data_decoder.cuda(), rel_type_onehot.cuda()
            data_decoder = data_decoder.contiguous()

            data_decoder, rel_type_onehot = Variable(data_decoder), Variable(rel_type_onehot)
            
            target = data_decoder[:, :, 1:, :] # dimensions are [batch, particle, time, state]
            output = decoder(data_decoder, rel_type_onehot, rel_rec, rel_send, 1)

            if args.plot:
                import matplotlib.pyplot as plt
                output_plot = decoder(data_decoder, rel_type_onehot, rel_rec, rel_send, 49)

                from trajectory_plot import draw_lines
                for i in range(args.batch_size):
                    fig = plt.figure(figsize=(7, 7))
                    ax = fig.add_axes([0, 0, 1, 1])
                    xmin_t, ymin_t, xmax_t, ymax_t = draw_lines( target, i, linestyle=':', alpha=0.6 )
                    xmin_o, ymin_o, xmax_o, ymax_o = draw_lines( output_plot.detach().numpy(), i, linestyle='-' )

                    ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
                    ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    plt.show()


            loss_nll = nll_gaussian(output, target, args.var)
            loss_nll_var = nll_gaussian_var(output, target, args.var)

            output_M = decoder(data_decoder, rel_type_onehot, rel_rec, rel_send, args.prediction_steps)
            loss_nll_M = nll_gaussian(output_M, target, args.var)
            loss_nll_M_var = nll_gaussian_var(output_M, target, args.var)

            mse_val.append(F.mse_loss(output_M, target).data.item())
            nll_val.append(loss_nll.data.item())
            nll_var_val.append(loss_nll_var.data.item())

            nll_M_val.append(loss_nll_M.data.item())
            nll_M_var_val.append(loss_nll_M_var.data.item())


    print('Epoch: {:03d}'.format(epoch),
          'time: {:.1f}s'.format(time.time() - t),
          'nll_trn: {:.2f}'.format(np.mean(nll_train)),
          'mse_trn: {:.10f}'.format(np.mean(mse_train)),
          'nll_val: {:.2f}'.format(np.mean(nll_M_val)),
          'mse_val: {:.10f}'.format(np.mean(mse_val))
          )

    print('Epoch: {:03d}'.format(epoch),
          'time: {:.1f}s'.format(time.time() - t),
          'nll_trn: {:.2f}'.format(np.mean(nll_train)),
          'mse_trn: {:.10f}'.format(np.mean(mse_train)),
          'nll_val: {:.2f}'.format(np.mean(nll_M_val)),
          'mse_val: {:.10f}'.format(np.mean(mse_val)),
          file=log)

    if epoch == 0:
        labels =  [ 'epoch', 'nll trn', 'mse train', 'nll var trn' ]
        labels += [ 'nll val', 'nll M val', 'mse val', 'nll var val', 'nll M var val' ]
        csv_writer.writerow( labels )

    csv_writer.writerow( [epoch, np.mean(nll_train), np.mean(mse_train), np.mean(nll_var_train)] +
                         [np.mean(nll_val), np.mean(nll_M_val), np.mean(mse_val)] +
                         [np.mean(nll_var_val), np.mean(nll_M_var_val)]
                        )

    log.flush()
    if args.save_folder and np.mean(nll_M_val) < best_val_loss:
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
    return np.mean(nll_M_val)


def test():
    t = time.time()
    nll_test = []
    nll_var_test = []
    mse_1_test = []
    mse_10_test = []
    mse_20_test = []
    mse_static = []

    nll_M_test = []
    nll_M_var_test = []

    decoder.eval()
    if not args.cuda:
        decoder.load_state_dict(torch.load(decoder_file,map_location='cpu'))
    else:
        decoder.load_state_dict(torch.load(decoder_file))

    for batch_idx, (data, relations) in enumerate(test_loader):
        with torch.no_grad():

            if args.full_graph:
                zeros = torch.zeros([data.size(0), rel_rec.size(0)])
                ones = torch.ones([data.size(0), rel_rec.size(0)])
                if args.NRI:
                    stack = [ ones ] + [ zeros for _ in range(edge_types-1) ]
                    rel_type_onehot = torch.stack(stack, -1)
                elif args.sigmoid:
                    stack = [ ones for _ in range(args.num_factors) ]
                    rel_type_onehot = torch.stack(stack, -1)
                else:
                    stack = []
                    for i in range(len(args.edge_types_list)):
                        stack += [ ones ] + [ zeros for _ in range(args.edge_types_list[i]-1) ]
                    rel_type_onehot = torch.stack(stack, -1)

            else:
                if args.NRI:
                    rel_type_onehot = torch.FloatTensor(data.size(0), rel_rec.size(0), edge_types)
                    rel_type_onehot.zero_()
                    rel_type_onehot.scatter_(2, relations.view(data.size(0), -1, 1), 1)
                elif args.sigmoid:
                    rel_type_onehot = relations.transpose(1,2).type(torch.FloatTensor)
                else:
                    rel_type_onehot = [ torch.FloatTensor(data.size(0), rel_rec.size(0), types) for types in args.edge_types_list ]
                    rel_type_onehot = [ rel.zero_() for rel in rel_type_onehot ]
                    rel_type_onehot = [ rel_type_onehot[i].scatter_(2, relations[:,i,:].view(data.size(0), -1, 1), 1) for i in range(len(rel_type_onehot)) ]
                    rel_type_onehot = torch.cat( rel_type_onehot, dim=-1 )

            data_decoder = data[:, :, -args.timesteps:, :]

            if args.cuda:
                data_decoder, rel_type_onehot = data_decoder.cuda(), rel_type_onehot.cuda()
            data_decoder = data_decoder.contiguous()

            data_decoder, rel_type_onehot = Variable(data_decoder), Variable(rel_type_onehot)
            
            target = data_decoder[:, :, 1:, :] # dimensions are [batch, particle, time, state]
            output = decoder(data_decoder, rel_type_onehot, rel_rec, rel_send, 1)


            if args.plot:
                import matplotlib.pyplot as plt
                output_plot = decoder(data_decoder, rel_type_onehot, rel_rec, rel_send, 49)
                from trajectory_plot import draw_lines
                for i in range(args.batch_size):
                    fig = plt.figure(figsize=(7, 7))
                    ax = fig.add_axes([0, 0, 1, 1])
                    xmin_t, ymin_t, xmax_t, ymax_t = draw_lines( target, i, linestyle=':', alpha=0.6 )
                    xmin_o, ymin_o, xmax_o, ymax_o = draw_lines( output_plot.detach().numpy(), i, linestyle='-' )

                    ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
                    ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    #plt.savefig(os.path.join(args.load_folder,str(i)+'_pred_and_true_.png'), dpi=300)
                    plt.show()
                   

            loss_nll = nll_gaussian(output, target, args.var)
            loss_nll_var = nll_gaussian_var(output, target, args.var)

            output_M = decoder(data_decoder, rel_type_onehot, rel_rec, rel_send, args.prediction_steps)
            loss_nll_M = nll_gaussian(output_M, target, args.var)

            output_10 = decoder(data_decoder, rel_type_onehot, rel_rec, rel_send, 10)
            output_20 = decoder(data_decoder, rel_type_onehot, rel_rec, rel_send, 20)
            mse_1_test.append(F.mse_loss(output, target).data.item())
            mse_10_test.append(F.mse_loss(output_10, target).data.item())
            mse_20_test.append(F.mse_loss(output_20, target).data.item())

            static = F.mse_loss(data_decoder[:, :, :-1, :], data_decoder[:, :, 1:, :])
            mse_static.append(static.data.item())

            nll_test.append(loss_nll.data.item())
            nll_var_test.append(loss_nll_var.data.item())
            nll_M_test.append(loss_nll_M.data.item())


    print('--------------------------------')
    print('------------Testing-------------')
    print('--------------------------------')
    print('nll_test: {:.2f}'.format(np.mean(nll_test)),
          'nll_M_test: {:.2f}'.format(np.mean(nll_M_test)),
          'mse_1_test: {:.10f}'.format(np.mean(mse_1_test)),
          'mse_10_test: {:.10f}'.format(np.mean(mse_10_test)),
          'mse_20_test: {:.10f}'.format(np.mean(mse_20_test)),
          'mse_static: {:.10f}'.format(np.mean(mse_static)),
          'time: {:.1f}s'.format(time.time() - t))
    print('--------------------------------', file=log)
    print('------------Testing-------------', file=log)
    print('--------------------------------', file=log)
    print('nll_test: {:.2f}'.format(np.mean(nll_test)),
          'nll_M_test: {:.2f}'.format(np.mean(nll_M_test)),
          'mse_1_test: {:.10f}'.format(np.mean(mse_1_test)),
          'mse_10_test: {:.10f}'.format(np.mean(mse_10_test)),
          'mse_20_test: {:.10f}'.format(np.mean(mse_20_test)),
          'mse_static: {:.10f}'.format(np.mean(mse_static)),
          'time: {:.1f}s'.format(time.time() - t),
          file=log)
    log.flush()


# Train model
if not args.test:
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    for epoch in range(args.epochs):
        val_loss = train(epoch, best_val_loss)
        if val_loss < best_val_loss:
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
