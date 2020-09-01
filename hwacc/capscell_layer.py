
from . import generic_layer 
from .caps_layer import caps_layer
from .convcaps3d_layer import convcaps3d_layer 

class CapsCellLayer(generic_layer.GenericLayer):
    def __init__(self, sa_rows, sa_cols, pe_width, pe_stages, 
        insize, inchannels, incapsules, kernsize, outsize, 
        outchannels, outcapsules, threedskip, period=2, pe_out_bw = 25):

        generic_layer.GenericLayer.__init__(self, sa_rows = sa_rows, sa_cols = sa_cols, pe_width = pe_width, pe_stages = pe_stages, period=period, pe_out_bw=pe_out_bw)

        self.insize = insize
        self.inchannels = inchannels
        self.incapsules = incapsules
        self.kernsize = kernsize
        self.outsize = outsize
        self.outchannels = outchannels
        self.outcapsules = outcapsules
        self.threedskip = threedskip

    def get_memory(self): #memory footprint
        if self.threedskip==0:
            ret_mem = 4*caps_layer.get_memory(self)
        elif self.threedskip==1:
            ret_mem = 3*caps_layer.get_memory(self) + convcaps3d_layer.get_memory(self)
        return ret_mem

    def get_cycles(self): #latency
        if self.threedskip==0:
            ret_cycles = 3*caps_layer.get_cycles(self)
        elif self.threedskip==1:
            ret_cycles = max(3*caps_layer.get_cycles(self), convcaps3d_layer.get_cycles(self))
        return ret_cycles

    def get_sums_per_out(self):
        if self.threedskip==0:
            ret_spo = 3*caps_layer.get_sums_per_out(self)
        elif self.threedskip==1:
            ret_spo = 3*caps_layer.get_sums_per_out(self) + convcaps3d_layer.get_sums_per_out(self) 
        return ret_spo

    def get_data_per_weight(self):        
        if self.threedskip==0:
            ret_dpw = 3*caps_layer.get_data_per_weight(self)
        elif self.threedskip==1:
            ret_dpw = 3*caps_layer.get_data_per_weight(self)+ convcaps3d_layer.get_data_per_weight(self)
        return ret_dpw

    def __str__(self):
        return "CapsCellLayer(%s)" % ",".join(map(str, [
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

