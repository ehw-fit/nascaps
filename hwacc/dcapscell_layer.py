# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 18:41:34 2020

@author: MA
"""


from . import generic_layer
from .sum_update_layer import sum_update_layer

class DCapsCellLayer(generic_layer.GenericLayer):
    
    def __init__(self, sa_rows, sa_cols, pe_width, pe_stages,
        insize, inchannels, incapsules, kernsize, outsize, 
        outchannels, outcapsules, period=2, pe_out_bw = 25):
        generic_layer.GenericLayer.__init__(self, sa_rows = sa_rows, sa_cols = sa_cols, pe_width = pe_width, pe_stages = pe_stages, period=period, pe_out_bw=pe_out_bw)
        raise NotImplementedError("invalid implementation")
        self.insize = insize
        self.inchannels = inchannels
        self.incapsules = incapsules
        self.kernsize = kernsize
        self.outsize = outsize
        self.outchannels = outchannels
        self.outcapsules = outcapsules

    def get_memory(self): #memory footprint
        ret_mem = 6*sum_update_layer.get_memory(self)
        return ret_mem

    def get_cycles(self): #latency
        ret_cycles = 6*sum_update_layer.get_cycles(self)
        return ret_cycles

    def get_sums_per_out(self):
        return 6*sum_update_layer.get_sums_per_out(self)

    def get_data_per_weight(self):        
        return 6*sum_update_layer.get_data_per_weight(self)


    def __str__(self):
        return "DCapsCellLayer(%s)" % ",".join(map(str, [
                self.sa_rows, 
                self.sa_cols, 
                self.pe_width, 
                self.pe_stages,
                self.insize, 
                self.inchannels, 
                self.incapsules, 
                self.kernsize, 
                self.outsize, 
                self.outchannels, 
                self.outcapsules,        
                self.period, 
                self.pe_out_bw]))

