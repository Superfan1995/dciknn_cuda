'''
Code for Fast k-Nearest Neighbour Search via Prioritized DCI

This code implements the method described in the Prioritized DCI paper,
which can be found at https://arxiv.org/abs/1703.00440

This file is a part of the Dynamic Continuous Indexing reference
implementation.


This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Copyright (C) 2020    Ke Li, Shichong Peng, Mehran Aghabozorgi
'''
from dciknn_cuda import DCI, MDCI
import torch
import random
import datetime

random_seed = 1
torch.manual_seed(random_seed)

def gen_data(ambient_dim, intrinsic_dim, num_points, num_heads):
    latent_data = torch.randn((num_heads, num_points, intrinsic_dim))
    transformation = torch.randn((num_heads, intrinsic_dim, ambient_dim)) 
    data = torch.matmul(latent_data, transformation)
    return data     # num_points x ambient_dim


def main():
    assert torch.cuda.is_available()
    device = torch.device('cuda:1')

    #############################################################################################################################################
    #                                                                                                                                           #
    # Data Generation Hyperparameters                                                                                                           #
    #                                                                                                                                           #
    #############################################################################################################################################
    dim = 10
    num_pts = 300
    num_queries = 50
    dim = 100
    #num_pts = 3000
    #num_queries = 500
    #num_heads = 1
    # dim = 100
    # num_pts = 3000
    # num_queries = 500

    intrinsic_dim = 40
    
    data_and_queries = gen_data(dim, intrinsic_dim, num_pts + num_queries, num_heads)

    #data = data_and_queries[:, :num_pts, :].detach().clone().to(device)
    #query = data_and_queries[:, num_pts:, :].detach().clone().to(device)

    #############################################################################################################################################
    #                                                                                                                                           #
    # Problem Hyperparameter                                                                                                                    #
    #                                                                                                                                           #
    #############################################################################################################################################
    num_neighbours = 10  # The k in k-NN

    #############################################################################################################################################
    #                                                                                                                                           #
    # DCI Hyperparameters                                                                                                                       #
    #                                                                                                                                           #
    #############################################################################################################################################
    block_size = 2
    thread_size = 10
    num_comp_indices = 2
    num_simp_indices = 10
    num_outer_iterations = 5000
    #block_size = 100
    #thread_size = 10
    #num_comp_indices = 2
    #num_simp_indices = 10
    #num_outer_iterations = 5000

    # initialize the DCI instance
    for i in range(1):
        #a = datetime.datetime.now()
        #dci_db = MDCI(dim, num_comp_indices, num_simp_indices, block_size, thread_size, devices=[0, 1])

        #dci_db.add(data)
        # Query
        #indices, dists = dci_db.query(query, num_neighbours, num_outer_iterations)
        #print("Nearest Indices:", indices)
        #print("Indices Distances:", dists)
        #dci_db.clear()
        #b = datetime.datetime.now()
        #print(b-a)

        data = data_and_queries[:, :num_pts, :].detach().clone().to(0)
        query = data_and_queries[:, num_pts:, :].detach().clone().to(0)

        a = datetime.datetime.now()
        dci_db = DCI(dim, num_heads, num_comp_indices, num_simp_indices, block_size, thread_size, device=0)

        dci_db.add(data)
        # Query
        dci_db.query(query, num_neighbours, num_outer_iterations)
        #print("Nearest Indices:", indices)
        #print("Indices Distances:", dists)
        #dci_db.clear()
        #b = datetime.datetime.now()
        #print(b-a)

        #print(len(data))
        #print(len(query))

if __name__ == '__main__':
    main()
