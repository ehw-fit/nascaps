
from . import generic_layer
import math

class ClassLayer(generic_layer.GenericLayer):
    def __init__(self, sa_rows, sa_cols, pe_width, pe_stages, 
        insize, inchannels, incapsules, kernsize, outsize, 
        outchannels, outcapsules, period=2, pe_out_bw = 25):

        generic_layer.GenericLayer.__init__(self, sa_rows = sa_rows, sa_cols = sa_cols, pe_width = pe_width, pe_stages = pe_stages, period=period, pe_out_bw=pe_out_bw)

        self.insize = insize
        self.inchannels = inchannels
        self.incapsules = incapsules
        self.kernsize = kernsize
        self.outsize = outsize
        self.outchannels = outchannels
        self.outcapsules = outcapsules

    def get_memory(self):
        return (self.inchannels*self.kernsize*self.kernsize+1)*self.outchannels*self.outcapsules*self.incapsules # weights
 
    def get_cycles(self):
        out_data = self.outsize*self.outsize*self.outchannels*self.outcapsules # L2
        weights = (self.inchannels*self.kernsize*self.kernsize+1)*self.outchannels*self.outcapsules*self.incapsules # weights
        sums_per_out = (self.kernsize*self.kernsize+1)*self.inchannels*self.incapsules # sums_per_out
        data_per_weight = 1 # data_per_weight
        w_load_cycles = self.sa_rows+self.pe_stages-1 # w_load_cycles
        w_loads = math.ceil(weights/self.sa_cols/min((self.sa_rows*self.pe_width), sums_per_out)) # w_loads
        cycles = w_load_cycles*w_loads+data_per_weight
        return cycles

        
    def get_sums_per_out(self):
        return (self.kernsize*self.kernsize+1)*self.inchannels*self.incapsules # sums_per_out

    def get_data_per_weight(self):        
        return 1 # data_per_weight


    def __str__(self):
        return "ClassLayer(%s)" % ",".join(map(str, [
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
