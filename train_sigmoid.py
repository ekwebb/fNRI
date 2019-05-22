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
parser.add_argument('--lr-decay', type=int, default=1000,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay value for L2 regularisation in Adam optimiser')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--patience', type=int, default=500,
                    help='Early stopping patience')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dont-split-data', action='store_true', default=False,
                    help='Whether to not split training and validation data into two parts')
parser.add_argument('--split-enc-only', action='store_true', default=False,
                    help='Whether to give the encoder the first half of trajectories \
                          and the decoder the whole of the trajectories')

## arguments related to loss function ##
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')   ## this is only here to rescale mse for NRI and fNRI comparison

## arguments related to weight and bias initialisation ##
parser.add_argument('--seed', type=int, default=1, 
                    help='Random seed.')
parser.add_argument('--encoder-init-type',type=str, default='xavier_normal',
                    help='The type of weight initialization to use in the encoder')
parser.add_argument('--decoder-init-type',type=str, default='default',
                    help='The type of weight initialization to use in the decoder')
parser.add_argument('--encoder-bias-scale',type=float, default=0.1,
                    help='The type of weight initialization to use in the encoder')

## arguments related to changing the model ##
parser.add_argument('--num-factors', type=int, default=2,
                    help='The number of factors to use')
parser.add_argument('--split-point', type=int, default=0,
                    help='The point at which factor graphs are split up in the encoder' )
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--sigmoid-sharpness', type=float, default=1.,
                    help='Coefficient in the power of the sigmoid function')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Round edges to integers while retaining non-rounded gradients during training.')

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

print(args)

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

    perm_csv_file = os.path.join(save_folder, 'perm_csv.csv')
    perm_csv = open(perm_csv_file, 'w')
    perm_writer = csv.writer(perm_csv, delimiter=',')
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data_fNRI(
                                                        args.batch_size, args.sim_folder, 
                                                        shuffle=True,
                                                        data_folder=args.data_folder)


# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms) 
rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)


if args.encoder == 'mlp':
    encoder = MLPEncoder_sigmoid(args.timesteps * args.dims, args.encoder_hidden,
                                   args.num_factors,args.encoder_dropout,
                                   split_point=args.split_point)

elif args.encoder == 'random':
    encoder = RandomEncoder(args.num_factors, args.cuda)

elif args.encoder == 'ones':
    encoder = OnesEncoder(args.num_factors, args.cuda)

if args.decoder == 'mlp':
    decoder = MLPDecoder_sigmoid(n_in_node=args.dims,
                                 num_factors=args.num_factors,
                                 msg_hid=args.decoder_hidden,
                                 msg_out=args.decoder_hidden,
                                 n_hid=args.decoder_hidden,
                                 do_prob=args.decoder_dropout,
                                 init_type=args.decoder_init_type)

elif args.decoder == 'stationary':
    decoder = StationaryDecoder()

elif args.decoder == 'velocity':
    decoder = VelocityStepDecoder()

if args.load_folder:
    print('Loading model from: '+args.load_folder)
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    if not args.cuda:
        encoder.load_state_dict(torch.load(encoder_file,map_location='cpu'))
        decoder.load_state_dict(torch.load(decoder_file,map_location='cpu'))
    else:
        encoder.load_state_dict(torch.load(encoder_file))
        decoder.load_state_dict(torch.load(decoder_file))
    args.save_folder = False

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)


if args.cuda:
    encoder.cuda()
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

    kl_train = []
    kl_list_train = []
    kl_var_list_train = []

    acc_train = []
    perm_train = []
    acc_blocks_train = []
    acc_var_train = []
    acc_var_blocks_train = []
    
    KLb_train = []
    KLb_blocks_train = []

    encoder.train()
    decoder.train()
    scheduler.step()
    if not args.plot:
        for batch_idx, (data, relations) in enumerate(train_loader): # relations are the ground truth interactions graphs
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(data), Variable(relations)

            if args.dont_split_data:
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data[:, :, :args.timesteps, :].contiguous()
            elif args.split_enc_only:
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data
            else:
                assert (data.size(2) - args.timesteps) >= args.timesteps
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data[:, :, -args.timesteps:, :].contiguous()

            optimizer.zero_grad()

            logits = encoder(data_encoder, rel_rec, rel_send)

            # dim of logits, edges and prob are [batchsize, N^2-N, edgetypes] where N = no. of particles
            
            edges = my_sigmoid(logits, hard=args.hard, sharpness=args.sigmoid_sharpness)

            loss_kl = 0
            loss_kl_split = [ 0 ]
            loss_kl_var_split = [ 0 ]
            
            KLb_train.append( 0 )
            KLb_blocks_train.append([0])

            acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_sigmoid(edges, relations)

            target = data_decoder[:, :, 1:, :] # dimensions are [batch, particle, time, state]
            output = decoder(data_decoder, edges, rel_rec, rel_send, args.prediction_steps)

            loss_nll = nll_gaussian(output, target, args.var)
            loss_nll_var = nll_gaussian_var(output, target, args.var)

            loss = F.mse_loss(output, target)

            perm_train.append(perm)
            acc_train.append(acc_perm)
            acc_blocks_train.append(acc_blocks)
            acc_var_train.append(acc_var)
            acc_var_blocks_train.append(acc_var_blocks)

            loss.backward() 
            optimizer.step()

            mse_train.append(loss.data.item()) 
            nll_train.append(loss_nll.data.item())
            nll_var_train.append(loss_nll_var.data.item())


    nll_val = []
    nll_var_val = []
    mse_val = []

    kl_val = []
    kl_list_val = []
    kl_var_list_val = []
    
    acc_val = []
    acc_blocks_val = []
    acc_var_val = []
    acc_var_blocks_val = []
    perm_val = []

    KLb_val = []
    KLb_blocks_val = [] # KL between blocks list

    nll_M_val = []
    nll_M_var_val = []

    encoder.eval()
    decoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        with torch.no_grad():
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()

            if args.dont_split_data:
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data[:, :, :args.timesteps, :].contiguous()
            elif args.split_enc_only:
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data
            else:
                assert (data.size(2) - args.timesteps) >= args.timesteps
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data[:, :, -args.timesteps:, :].contiguous()

            # dim of logits, edges are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
            logits = encoder(data_encoder, rel_rec, rel_send)

            edges = my_sigmoid(logits, hard=args.hard, sharpness=args.sigmoid_sharpness)

            loss_kl = 0
            loss_kl_split = [ 0 ]
            loss_kl_var_split = [ 0 ]
            
            KLb_train.append( 0 )
            KLb_blocks_train.append([0])

            acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_sigmoid(edges, relations)

            target = data_decoder[:, :, 1:, :] # dimensions are [batch, particle, time, state]
            output = decoder(data_decoder, edges, rel_rec, rel_send, 1)

            if args.plot:
                import matplotlib.pyplot as plt
                output_plot = decoder(data_decoder, edges, rel_rec, rel_send, 49)

                acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_sigmoid_batch(edges, relations)

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
                    block_names = [ str(j) for j in range(args.num_factors) ]
                    acc_text = [ 'layer ' + block_names[j] + ' acc: {:02.0f}%'.format(100*acc_blocks_batch[i,j]) 
                                 for j in range(acc_blocks_batch.shape[1]) ]
                    acc_text = ', '.join(acc_text)
                    plt.text( 0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes )
                    plt.show()


            loss_nll = nll_gaussian(output, target, args.var)
            loss_nll_var = nll_gaussian_var(output, target, args.var)

            output_M = decoder(data_decoder, edges, rel_rec, rel_send, args.prediction_steps)
            loss_nll_M = nll_gaussian(output_M, target, args.var)
            loss_nll_M_var = nll_gaussian_var(output_M, target, args.var)
            
            perm_val.append(perm)
            acc_val.append(acc_perm)
            acc_blocks_val.append(acc_blocks)
            acc_var_val.append(acc_var)
            acc_var_blocks_val.append(acc_var_blocks)

            mse_val.append(F.mse_loss(output_M, target).data.item())
            nll_val.append(loss_nll.data.item())
            nll_var_val.append(loss_nll_var.data.item())

            nll_M_val.append(loss_nll_M.data.item())
            nll_M_var_val.append(loss_nll_M_var.data.item())

    print('Epoch: {:03d}'.format(epoch),
          'perm_val: '+str( np.around(np.mean(np.array(perm_val),axis=0),4 ) ),
          'time: {:.1f}s'.format(time.time() - t))
    print('nll_trn: {:.2f}'.format(np.mean(nll_train)),
          'mse_trn: {:.10f}'.format(np.mean(mse_train)),
          'acc_trn: {:.5f}'.format(np.mean(acc_train)),
          'acc_b_trn: '+str( np.around(np.mean(np.array(acc_blocks_train),axis=0),4 ) )
          ) 
    print('nll_val: {:.2f}'.format(np.mean(nll_M_val)),
          'mse_val: {:.10f}'.format(np.mean(mse_val)),
          'acc_val: {:.5f}'.format(np.mean(acc_val)),
          'acc_b_val: '+str( np.around(np.mean(np.array(acc_blocks_val),axis=0),4 ) ) 
          ) 
    print('Epoch: {:03d}'.format(epoch),
          'perm_val: '+str( np.around(np.mean(np.array(perm_val),axis=0),4 ) ),
          'time: {:.1f}s'.format(time.time() - t),
          file=log )
    print('nll_trn: {:.2f}'.format(np.mean(nll_train)),
          'mse_trn: {:.10f}'.format(np.mean(mse_train)),
          'acc_trn: {:.5f}'.format(np.mean(acc_train)),
          'acc_b_trn: '+str( np.around(np.mean(np.array(acc_blocks_train),axis=0),4 ) ),
          file=log ) 
    print('nll_val: {:.2f}'.format(np.mean(nll_val)),
          'nll_M_val: {:.2f}'.format(np.mean(nll_M_val)),
          'mse_val: {:.10f}'.format(np.mean(mse_val)),
          'acc_val: {:.5f}'.format(np.mean(acc_val)),
          'acc_b_val: '+str( np.around(np.mean(np.array(acc_blocks_val),axis=0),4 ) ),
          file=log ) 
    if epoch == 0:
        labels =  [ 'epoch', 'nll trn', 'mse train', 'nll var trn', 'acc trn' ]
        labels += [ 'b'+str(i)+' acc trn' for i in range( args.num_factors ) ]
        labels += [ 'acc var trn']  + [ 'b'+str(i)+' acc var trn' for i in range( args.num_factors ) ]
        labels += [ 'nll val', 'nll M val', 'mse val', 'acc val' ]
        labels += [ 'b'+str(i)+' acc val' for i in range( args.num_factors ) ]
        labels += [ 'nll var val', 'nll M var val' ]
        labels += [ 'acc var val']  + [ 'b'+str(i)+' acc var val' for i in range( args.num_factors ) ]
        csv_writer.writerow( labels )

        labels = [ 'trn '+str(i) for i in range(len(perm_train[0])) ]
        labels += [ 'val '+str(i) for i in range(len(perm_val[0])) ]
        perm_writer.writerow( labels )

    csv_writer.writerow( [epoch, np.mean(nll_train), np.mean(mse_train), np.mean(nll_var_train), np.mean(acc_train)] +
                         list(np.mean(np.array(acc_blocks_train),axis=0)) +
                         [np.mean(acc_var_train)] + list(np.mean(np.array(acc_var_blocks_train),axis=0)) +
                         [np.mean(nll_val), np.mean(nll_M_val), np.mean(mse_val), np.mean(acc_val) ] +
                         list(np.mean(np.array(acc_blocks_val),axis=0)) + 
                         [np.mean(nll_var_val), np.mean(nll_M_var_val)] +
                         [np.mean(acc_var_val)] + list(np.mean(np.array(acc_var_blocks_val),axis=0))
                        )
    perm_writer.writerow( list(np.mean(np.array(perm_train),axis=0)) +
                          list(np.mean(np.array(perm_val),axis=0)) 
                          )

    log.flush()
    if args.save_folder and np.mean(nll_M_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
    return np.mean(nll_M_val)


def test():
    nll_test = []
    nll_var_test = []
    
    acc_test = []
    acc_blocks_test = []
    acc_var_test = []
    acc_var_blocks_test = []
    perm_test = []

    mse_1_test = []
    mse_10_test = []
    mse_20_test = []

    nll_M_test = []
    nll_M_var_test = []

    encoder.eval()
    decoder.eval()
    if not args.cuda:
        encoder.load_state_dict(torch.load(encoder_file,map_location='cpu'))
        decoder.load_state_dict(torch.load(decoder_file,map_location='cpu'))
    else:
        encoder.load_state_dict(torch.load(encoder_file))
        decoder.load_state_dict(torch.load(decoder_file))

    for batch_idx, (data, relations) in enumerate(test_loader):
        with torch.no_grad():
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()

            assert (data.size(2) - args.timesteps) >= args.timesteps
            data_encoder = data[:, :, :args.timesteps, :].contiguous()
            data_decoder = data[:, :, -args.timesteps:, :].contiguous()

            # dim of logits, edges and prob are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
            logits = encoder(data_encoder, rel_rec, rel_send)
            edges = edges = my_sigmoid(logits, hard=args.hard, sharpness=args.sigmoid_sharpness)

            acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_sigmoid(edges, relations)

            target = data_decoder[:, :, 1:, :] # dimensions are [batch, particle, time, state]
            output = decoder(data_decoder, edges, rel_rec, rel_send, 1)

            if args.plot:
                import matplotlib.pyplot as plt
                output_plot = decoder(data_decoder, edges, rel_rec, rel_send, 49)

                output_plot_en = decoder(data_encoder, edges, rel_rec, rel_send, 49)
                from trajectory_plot import draw_lines

                acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_sigmoid_batch(edges, relations)

                for i in range(args.batch_size):
                    fig = plt.figure(figsize=(7, 7))
                    ax = fig.add_axes([0, 0, 1, 1])
                    xmin_t, ymin_t, xmax_t, ymax_t = draw_lines( target, i, linestyle=':', alpha=0.6 )
                    xmin_o, ymin_o, xmax_o, ymax_o = draw_lines( output_plot.detach().numpy(), i, linestyle='-' )

                    ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
                    ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    block_names = [str(j) for j in range(args.num_factors)]
                    acc_text = [ 'layer ' + block_names[j] + ' acc: {:02.0f}%'.format(100*acc_blocks_batch[i,j]) 
                                 for j in range(acc_blocks_batch.shape[1]) ]
                    acc_text = ', '.join(acc_text)
                    plt.text( 0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes )
                    #plt.savefig(os.path.join(args.load_folder,str(i)+'_pred_and_true_.png'), dpi=300)
                    plt.show()

                    
            loss_nll = nll_gaussian(output, target, args.var)
            loss_nll_var = nll_gaussian_var(output, target, args.var)
    
            output_10 = decoder(data_decoder, edges, rel_rec, rel_send, 10)
            output_20 = decoder(data_decoder, edges, rel_rec, rel_send, 20)
            mse_1_test.append(F.mse_loss(output, target).data.item())
            mse_10_test.append(F.mse_loss(output_10, target).data.item())
            mse_20_test.append(F.mse_loss(output_20, target).data.item())

            loss_nll_M = nll_gaussian(output_10, target, args.var)
            loss_nll_M_var = nll_gaussian_var(output_10, target, args.var)

            perm_test.append(perm)
            acc_test.append(acc_perm)
            acc_blocks_test.append(acc_blocks)
            acc_var_test.append(acc_var)
            acc_var_blocks_test.append(acc_var_blocks)

            nll_test.append(loss_nll.data.item())
            nll_var_test.append(loss_nll_var.data.item())
            nll_M_test.append(loss_nll_M.data.item())
            nll_M_var_test.append(loss_nll_M_var.data.item())


    print('--------------------------------')
    print('------------Testing-------------')
    print('--------------------------------')
    print('nll_test: {:.2f}'.format(np.mean(nll_test)),
          'nll_M_test: {:.2f}'.format(np.mean(nll_M_test)),
          'mse_1_test: {:.10f}'.format(np.mean(mse_1_test)),
          'mse_10_test: {:.10f}'.format(np.mean(mse_10_test)),
          'mse_20_test: {:.10f}'.format(np.mean(mse_20_test)),
          'acc_test: {:.5f}'.format(np.mean(acc_test)),
          'acc_var_test: {:.5f}'.format(np.mean(acc_var_test)),
          'acc_b_test: '+str( np.around(np.mean(np.array(acc_blocks_test),axis=0),4 ) ),
          'acc_var_b_test: '+str( np.around(np.mean(np.array(acc_var_blocks_test),axis=0),4 ) )
          )
    print('--------------------------------', file=log)
    print('------------Testing-------------', file=log)
    print('--------------------------------', file=log)
    print('nll_test: {:.2f}'.format(np.mean(nll_test)),
          'nll_M_test: {:.2f}'.format(np.mean(nll_M_test)),
          'mse_1_test: {:.10f}'.format(np.mean(mse_1_test)),
          'mse_10_test: {:.10f}'.format(np.mean(mse_10_test)),
          'mse_20_test: {:.10f}'.format(np.mean(mse_20_test)),
          'acc_test: {:.5f}'.format(np.mean(acc_test)),
          'acc_var_test: {:.5f}'.format(np.mean(acc_var_test)),
          'acc_b_test: '+str( np.around(np.mean(np.array(acc_blocks_test),axis=0),4 ) ),
          'acc_var_b_test: '+str( np.around(np.mean(np.array(acc_var_blocks_test),axis=0),4 ) ),
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
    perm_csv.close()
