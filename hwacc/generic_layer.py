from __future__ import print_function
import json, gzip
import os
class GenericLayer:
    systable = None
    parsed = {}
    memtable = None

    def __init__(self, sa_rows, sa_cols, pe_width, pe_stages, period, pe_out_bw = 25):
        
        self.sa_rows = sa_rows
        self.sa_cols = sa_cols
        self.pe_width = pe_width
        self.pe_stages = pe_stages
        self.period =  period
        self.pe_out_bw = pe_out_bw

        if not self.memtable:
            self.memtable = json.load(gzip.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/mem.json.gz")))
    

    def get_cycles(self):
        raise NotImplemented("This function is implemented in special layers types")

    def get_config_(self):
        # find the value in the table
        if not GenericLayer.systable:
            d = json.load(gzip.open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data/pe.json.gz")))
            GenericLayer.systable = [d[x] for x in d if "_same" not in x]
        
        period = self.period * 1

        if period not in GenericLayer.parsed:
            GenericLayer.parsed[period] = {
                "__all__": list(filter(lambda x: x["period"] == period, GenericLayer.systable))
            }

        if self.pe_out_bw not in GenericLayer.parsed[period]:
             GenericLayer.parsed[period][self.pe_out_bw] = {
                 "__all__": list(filter(lambda x: x["nout"] == self.pe_out_bw, GenericLayer.parsed[period]["__all__"]))
             }

        if self.pe_stages not in GenericLayer.parsed[period][self.pe_out_bw]:
             GenericLayer.parsed[period][self.pe_out_bw][self.pe_stages] = {
                 "__all__": list(filter(lambda x: x["stages"] == self.pe_stages, GenericLayer.parsed[period][self.pe_out_bw]["__all__"]))
             }
        # find the selected item in the data
        data = filter(lambda x: x["ninputs"] == self.pe_width, GenericLayer.parsed[period][self.pe_out_bw][self.pe_stages]["__all__"])
        data = list(data)

        if len(data) == 0:
            raise KeyError("Configuration of %s was not found in the table" % (str(self)) )

            
        if len(data) > 1:
            print(data)
            raise KeyError("Multiple configurations of %s was found in the table" % (str(self)) )
        
        return data[0]

    def get_power_pe_(self):
        try:
            return self.get_config_()["pdk45_pwr"] * self.sa_rows * self.sa_cols
        except KeyError:
            return float("NaN")

    def get_area_pe_(self):
        try:
            return self.get_config_()["pdk45_area"] * self.sa_rows * self.sa_cols
        except KeyError:
            return float("NaN")

    def get_power_mem_(self, bw):
        # power consumption of a memory cell
        nname = "p%02d_b%02d" % (self.period * 1, bw)
        mem = self.memtable[nname]
        assert mem["bw"] == bw
        assert mem["period"] == self.period * 1
        return mem["pdk45_pwr"]

    def get_area_mem_(self, bw):
        # power consumption of a memory cell
        nname = "p%02d_b%02d" % (self.period * 1, bw)
        mem = self.memtable[nname]
        assert mem["bw"] == bw
        assert mem["period"] == self.period * 1
        return mem["pdk45_area"]

    def get_sums_per_out(self):
        raise NotImplementedError()

    def get_data_per_weight(self):
        raise NotImplementedError()

    def get_power(self):
        if self.get_data_per_weight() == 1:
            mem_accum = self.sa_cols
        else:
            mem_accum = self.sa_cols * max(self.get_sums_per_out() - self.sa_rows * self.pe_width + 1, 1)

        return self.get_power_pe_() + mem_accum * self.get_power_mem_(self.pe_out_bw)

    def get_period(self):
        return self.period * 1e-9

    def get_area(self):
        mem_accum = self.sa_cols * self.sa_rows

        return self.get_area_pe_() + mem_accum * self.get_area_mem_(self.pe_out_bw)

    def __str__(self):
        return "GenericLayer(%s)" % ",".join(map(str, [self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, self.period, self.pe_out_bw]))


if __name__ == "__main__":
    l = GenericLayer(sa_rows = 5, sa_cols = 4, pe_width=87, pe_stages=1, period=2)
    print(l.get_config_())
    print(GenericLayer(sa_rows = 5, sa_cols = 4, pe_width=450, pe_stages=1, period=2).get_config_())
    l = GenericLayer(sa_rows = 82, sa_cols = 1, pe_width=1, pe_stages=1, period=2)
    print(l.get_config_())