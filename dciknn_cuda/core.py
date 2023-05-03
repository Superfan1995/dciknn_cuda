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

import torch
from _dci_cuda import _dci_new, _dci_add, _dci_query, _dci_clear, _dci_reset, _dci_free, _dci_multi_query, _dci_multi_head_query
# from _dci_cuda import _dci_new, _dci_add, _dci_query, _dci_clear, _dci_reset, _dci_free

from math import sqrt


class DCI(object):
    
    def __init__(self, num_heads, dim, num_comp_indices=2, num_simp_indices=7, bs=100, ts=10, device=0):
        
        if not torch.cuda.is_available():
            raise RuntimeError("DCI CUDA version requires GPU access, please check CUDA driver.")

        self._dim = dim
        self._num_heads = num_heads
        self._num_comp_indices = num_comp_indices
        self._num_simp_indices = num_simp_indices
        self._dci_inst = _dci_new(num_heads, dim, num_comp_indices, num_simp_indices, device)
        self._array = None
        self._block_size = bs
        self._thread_size = ts
        self.num_points = 0

    @property
    def dim(self):
        return self._dim
    
    @property
    def num_heads(self):
        return self._num_heads
        
    @property
    def num_comp_indices(self):
        return self._num_comp_indices
        
    @property
    def num_simp_indices(self):
        return self._num_simp_indices
            
    def _ensure_positive_integer(self, x):
        if not isinstance(x, int):
            raise TypeError("number must be an integer")
        elif x <= 0:
            raise ValueError("number must be positive")
    
    def _check_data(self, arr):
        if arr.shape[1] != self.dim:
            raise ValueError("mismatch between tensor dimension (%d) and the declared dimension of this DCI instance (%d)" % (arr.shape[1], self.dim))
        if arr.dtype != torch.float:
            raise TypeError("tensor must consist of double-precision floats")
        if not arr.is_contiguous():
            raise ValueError("the memory layout of tensor must be in row-major (C-order)")
        if not arr.is_cuda:
            raise TypeError("tensor must be a cuda tensor")

    def add(self, data):
        if self.num_points > 0:
            raise RuntimeError("DCI class does not support insertion of more than one tensor. Must combine all tensors into one tensor before inserting")
        self._check_data(data)
        self.num_points = (int) (data.shape[0] / self._num_heads)
        _dci_add(self._dci_inst, self._num_heads, self._dim, self.num_points, data.flatten(), self._block_size, self._thread_size)
        self._array = data
    
    # query is num_queries x dim, returns num_queries x num_neighbours
    def query(self, query, num_neighbours=-1, num_outer_iterations=5000, blind=False):
        if len(query.shape) < 2:
            _query = query.unsqueeze(0)
        else:
            _query = query
        self._check_data(_query)
        if num_neighbours < 0:
            num_neighbours = self.num_points
        self._ensure_positive_integer(num_neighbours)
        max_num_candidates = 10 * num_neighbours
        # num_queries x num_neighbours

        num_queries = (int) (_query.shape[0] / self._num_heads)
        _query_result = _dci_query(self._dci_inst, self._num_heads, self._dim, num_queries, _query.flatten(), num_neighbours, blind, num_outer_iterations, max_num_candidates, self._block_size, self._thread_size)

        half = _query_result.shape[0] // 2
        return _query_result[:half].reshape(_query.shape[0], -1), _query_result[half:].reshape(_query.shape[0], -1)
    
    def clear(self):
        _dci_clear(self._dci_inst)
        self.num_points = 0
        self._array = None
    
    def reset(self):
        _dci_reset(self._dci_inst)
        self.num_points = 0
        self._array = None

    def free(self):
        _dci_free(self._dci_inst)
        self.num_points = 0
        self._array = None

# noting
# currently not working properly
class MDCI(object):
    def __init__(self, num_heads, dim, num_comp_indices=2, num_simp_indices=7, bs=100, ts=10, devices=[0]):
        # if len(devices) < 2:
        #     raise RuntimeError("You should specify at least two GPU for multi-GPU DCI to work")
        
        self._num_heads = num_heads
        self._dim = dim
        self._num_comp_indices = num_comp_indices
        self._num_simp_indices = num_simp_indices
        self._bs = bs
        self._ts = ts

        self.devices = devices
        self.num_devices = len(devices)
        self.dcis = []
        self.data_per_device = 0 # the number of data points assign to each device (note: for single head situation)
        self.head_per_device = 0 # the number of head assign to each device
        self.head_per_device_list = []
        self.num_points = 0
    
    # need consider the number of points
    def add(self, data):

        if (self._num_heads == 1):
            self.dcis = [DCI(self._num_heads, self._dim, self._num_comp_indices, self._num_simp_indices, self._bs, self._ts, dev) for dev in self.devices]
            self.data_per_device = data.shape[0] // self.num_devices + 1
            for dev_ind in range(self.num_devices):
                device = self.devices[dev_ind]
                cur_data = data[dev_ind * self.data_per_device: dev_ind * self.data_per_device + self.data_per_device].to(device)
                self.dcis[dev_ind].add(cur_data)

        else:
            self.head_per_device = self._num_heads // self.num_devices
            # number of data points in a single head
            self.num_points = data.shape[0] // self._num_heads
            for dev_ind in range(self.num_devices):
                # number of head assign to current device
                num_heads_device = min(self.head_per_device, self._num_heads - dev_ind * self.head_per_device)
                num_points_device = num_heads_device * self.num_points
                self.head_per_device_list.append(num_heads_device)
                
                device = self.devices[dev_ind]
                cur_data = data[dev_ind * self.head_per_device * self.num_points: dev_ind * self.head_per_device * self.num_points + num_points_device].to(device)

                new_dci = DCI(num_heads_device, self._dim, self._num_comp_indices, self._num_simp_indices, self._bs, self._ts, device)
                (self.dcis).append(new_dci)
                self.dcis[dev_ind].add(cur_data)

    def query(self, query, num_neighbours=-1, num_outer_iterations=5000, blind=False):
        dists = []
        nns = []
        if num_neighbours <= 0:
            raise RuntimeError('num_neighbours must be positive')

        if len(query.shape) < 2:
            _query = query.unsqueeze(0)
        else:
            _query = query
        _query = _query.detach().clone()

        max_num_candidates = 10 * num_neighbours

        num_queries = _query.shape[0] // self.dcis[0]._num_heads
        if (self._num_heads == 1):
            queries = [_query.to(self.devices[dev_ind]).flatten() for dev_ind in self.devices]
            res = _dci_multi_query([dc._dci_inst for dc in self.dcis], self.dcis[0]._num_heads, self.dcis[0]._dim, num_queries, queries, num_neighbours, blind, num_outer_iterations, max_num_candidates, self.dcis[0]._block_size, self.dcis[0]._thread_size)

            for ind, cur_res in enumerate(res):
                half = cur_res.shape[0] // 2
                cur_nns, cur_dist = cur_res[:half].reshape(num_queries, -1), cur_res[half:].reshape(num_queries, -1)
                cur_nns = cur_nns + self.data_per_device * ind
                dists.append(cur_dist.detach().clone().to(self.devices[0]))
                nns.append(cur_nns.detach().clone().to(self.devices[0]))

        else:
            queries = []
            for dev_ind in range(self.num_devices):
                cur_query = query[dev_ind * num_queries * self.head_per_device: dev_ind * num_queries * self.head_per_device + num_queries * self.head_per_device_list[dev_ind], :]
                queries.append(cur_query.to(self.devices[dev_ind]).flatten())
            res = _dci_multi_head_query([dc._dci_inst for dc in self.dcis], [head_per_device for head_per_device in self.head_per_device_list], self.dcis[0]._dim, num_queries, [query for query in queries], num_neighbours, blind, num_outer_iterations, max_num_candidates, self.dcis[0]._block_size, self.dcis[0]._thread_size)

            print("success query")

            for ind, cur_res in enumerate(res):
                half = cur_res.shape[0] // 2
                cur_num_queries = self.head_per_device_list[ind] * num_queries
                cur_nns, cur_dist = cur_res[:half].reshape(cur_num_queries, -1), cur_res[half:].reshape(cur_num_queries, -1)
                cur_nns = cur_nns + self.head_per_device * self.num_points * ind
                dists.append(cur_dist.detach().clone().to(self.devices[0]))
                nns.append(cur_nns.detach().clone().to(self.devices[0]))

        merged_dists = torch.cat(dists, dim=1)
        merged_nns = torch.cat(nns, dim=1)
        _, sort_indices = torch.sort(merged_dists, dim=1)
        sort_indices = sort_indices[:, :num_neighbours]
        return torch.gather(merged_nns, 1, sort_indices), torch.gather(merged_dists, 1, sort_indices)

    def clear(self):
        for dci in self.dcis:
            dci.clear()

    def reset(self):
        for dci in self.dcis:
            dci.reset()

    def free(self):
        for dci in self.dcis:
            dci.free()