"""
This code is based on https://github.com/ethanfetaya/NRI
(MIT licence)
"""
from synthetic_sim import *
import time
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--num-train', type=int, default=50000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=10000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=10000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--savefolder', type=str, default='springcharge_5',
                    help='name of folder to save everything in')
parser.add_argument('--sim-type', type=str, default='springcharge',
                    help='Type of simulation system')

args = parser.parse_args()
os.makedirs(args.savefolder)
par_file = open(os.path.join(args.savefolder,'sim_args.txt'),'w')
print(args, file=par_file)
par_file.flush()
par_file.close()

if args.sim_type == 'springcharge':
    sim = SpringChargeSim(noise_var=0.0, n_balls=args.n_balls, box_size=5.0)

elif args.sim_type == 'springchargequad':
    sim = SpringChargeQuadSim(noise_var=0.0, n_balls=args.n_balls, box_size=5.0)

elif args.sim_type == 'springquad':
    sim = SpringQuadSim(noise_var=0.0, n_balls=args.n_balls, box_size=5.0)

elif args.sim_type == 'springchargefspring':
    sim = SpringChargeFspringSim(noise_var=0.0, n_balls=args.n_balls, box_size=5.0)

np.random.seed(args.seed)

def generate_dataset(num_sims, length, sample_freq):
    loc_all = list()
    vel_all = list()
    edges_all = list()

    for i in range(num_sims):
        t = time.time()
        loc, vel, edges = sim.sample_trajectory(T=length, sample_freq=sample_freq)
        if i % 100 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)
        vel_all.append(vel)
        edges_all.append(edges)

    loc_all = np.stack(loc_all)
    vel_all = np.stack(vel_all)
    edges_all = np.stack(edges_all)

    return loc_all, vel_all, edges_all


print("Generating {} training simulations".format(args.num_train))
loc_train, vel_train, edges_train = generate_dataset(args.num_train, args.length, args.sample_freq)

np.save(os.path.join(args.savefolder,'loc_train.npy'), loc_train)
np.save(os.path.join(args.savefolder,'vel_train.npy'), vel_train)
np.save(os.path.join(args.savefolder,'edges_train.npy'), edges_train)

print("Generating {} validation simulations".format(args.num_valid))
loc_valid, vel_valid, edges_valid = generate_dataset(args.num_valid, args.length, args.sample_freq)

np.save(os.path.join(args.savefolder,'loc_valid.npy'), loc_valid)
np.save(os.path.join(args.savefolder,'vel_valid.npy'), vel_valid)
np.save(os.path.join(args.savefolder,'edges_valid.npy'), edges_valid)

print("Generating {} test simulations".format(args.num_test))
loc_test, vel_test, edges_test= generate_dataset(args.num_test, args.length_test, args.sample_freq)

np.save(os.path.join(args.savefolder,'loc_test.npy'), loc_test)
np.save(os.path.join(args.savefolder,'vel_test.npy'), vel_test)
np.save(os.path.join(args.savefolder,'edges_test.npy'), edges_test)
