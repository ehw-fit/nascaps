# this is example how to estimate the HW parameters

import sys
sys.path.append("..")
import hwacc

class hw_estimator:
    def __init__(self):
        self.layers = []
        # TODO discuss these parameters with Alberto; these parameters will be fixed during the evolution
        # for fixed parameters we will not focus on the     
        self.sa_rows = 16
        self.sa_cols = 16
        self.pe_width = 1
        self.pe_stages = 1
        self.period = 3

    def add_layer(self, name, layer):
        self.layers.append((name, layer))


    def get_energy(self):
        return sum([l.get_power() * l.get_cycles() * l.get_period() for _, l in self.layers])

    def get_latency(self):
        return sum([l.get_cycles() * l.get_period() for _, l in self.layers])

    def get_memory(self):
        return sum([l.get_memory() for _, l in self.layers])

    def __str__(self):
        return "<hwest>\n" + "\n".join(["  %s: %s" % (s, str(l).replace("(16,16,1,1,", "(").replace(",3,25)", ")")) for s, l in self.layers]) + "\n</hwest>"

    

    def parse_genotype(self, gene):
        self.layers = []
        # gene-specific part
        remaining=len(gene)-2 
        conv_index=1
        caps_index=1
        deepcaps_index=1
        resize_factor=gene[len(gene)-1][0]
       
        insize=gene[0][1]


        first=1
        
        for layer in gene:
            print(remaining)
        
            # Convolutional layers
            # conv = [0, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            if layer[0]==0 and len(layer)>1:
                self.add_layer("conv"+str(conv_index), hwacc.ConvLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=layer[1],
                                                                  inchannels=layer[2],incapsules=layer[3], kernsize=layer[4],
                                                                  outsize=layer[6], outchannels=layer[7],
                                                                  outcapsules=layer[8], period=self.period))
                conv_index=conv_index+1
                remaining=remaining-1
        
        
            # Primary Capsules layers
            # caps = [1, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            elif layer[0]==1 and remaining>1 and len(layer)>1:
                self.add_layer("caps"+str(caps_index), hwacc.CapsLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=layer[1],
                                                                       inchannels=layer[2],incapsules=layer[3], kernsize=layer[4],
                                                                       outsize=layer[6], outchannels=layer[7],
                                                                       outcapsules=layer[8], period=self.period))
                caps_index=caps_index+1
                remaining=remaining-1
                
                
            # Class Capsules layer
            # caps = [1, insize, inchannels, incapsules, kernsize, stride, outsize, outchannels, outcapsules]
            elif layer[0]==1 and remaining==1 and len(layer)>1:
                self.add_layer("Class", hwacc.ClassLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=layer[1], inchannels=layer[2], 
                                                         incapsules=layer[3], kernsize=layer[4], outsize=layer[6], outchannels=layer[7], outcapsules=layer[8],
                                                         period=self.period))
                for i in range(5):
                    self.add_layer("sum_%d" % i, hwacc.SumLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=layer[1], inchannels=layer[2], incapsules=layer[3], kernsize=layer[4], outsize=layer[6], outchannels=layer[7], outcapsules=layer[8], period=self.period))


                caps_index=caps_index+1
                remaining=remaining-1


            # DeepCaps Cells
            # d_caps = [2, 0, inchannels, incapsules, kernsize, stride, 0, outchannels, outcapsules]
            elif layer[0]==2 and remaining>1 and len(layer)>1:
                
                insize=layer[1]
                inchannels=layer[2]
                incapsules=layer[3]

                if remaining==2:
                    # 3x caps layer + 1x conv_caps_3d
                    # add also 
                    
                    for i in range(3):
                        self.add_layer(f"CapsCell{caps_index}_cl_{i}", hwacc.CapsLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=insize, inchannels=layer[2], 
                                                                                   incapsules=layer[3], kernsize=layer[4], outsize=layer[6], outchannels=layer[7], outcapsules=layer[8],
                                                                                   period=self.period))
                        if i==0:
                            insize=int(insize/2)


                    self.add_layer(f"CapsCell{caps_index}_c3d", hwacc.ConvCaps3D(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=insize, inchannels=layer[2], 
                                                                                incapsules=layer[3], kernsize=layer[4], outsize=layer[6], outchannels=layer[7], outcapsules=layer[8],
                                                                                period=self.period))

                else:
                    for i in range(4):
                        self.add_layer(f"CapsCell{caps_index}_cl_{i}", hwacc.CapsLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=insize, inchannels=layer[2], 
                                                                                   incapsules=incapsules, kernsize=layer[4], outsize=layer[6], outchannels=layer[7], outcapsules=layer[8],
                                                                                   period=self.period))
                        if i==0:
                            insize=int(insize/2)
                            incapsules=layer[8]

                caps_index=caps_index+1
                #count +=1
                deepcaps_index=deepcaps_index+1
                remaining=remaining-1

            
            #FlattenCaps
            elif layer[0]==2 and remaining==1 and len(layer)>1:

                for i in range(0, 3):
                    self.add_layer(f"FlattenCaps_ClassCaps_sum_{i}", hwacc.SumLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=layer[1], inchannels=layer[2], 
                                                                              incapsules=layer[3], kernsize=layer[4], outsize=layer[6], outchannels=layer[7], outcapsules=layer[8],
                                                                              period=self.period))
                    self.add_layer(f"FlattenCaps_ClassCaps_update_{i}", hwacc.UpdateLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=layer[1], inchannels=layer[2], 
                                                                              incapsules=layer[3], kernsize=layer[4], outsize=layer[6], outchannels=layer[7], outcapsules=layer[8],
                                                                              period=self.period))



                self.add_layer("ClassCaps_class", hwacc.ClassLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=layer[1], inchannels=layer[2], incapsules=layer[3], kernsize=layer[4], outsize=layer[6],
                                outchannels=layer[7], outcapsules=layer[8], period=self.period)) 

                remaining=remaining-1

            else:

                break


    def force_deepcaps(self):
        self.add_layer("Conv", hwacc.ConvLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                    insize=64, inchannels=3, incapsules=1, kernsize=3, outsize=64, outchannels=128, outcapsules=1, period=self.period))

        self.add_layer("ConvCaps2D_1", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=64, inchannels=32, incapsules=4, kernsize=3, outsize=32, outchannels=32, outcapsules=4, period=self.period))

        self.add_layer("ConvCaps2D_2", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=32, inchannels=32, incapsules=4, kernsize=3, outsize=32, outchannels=32, outcapsules=4, period=self.period))

        self.add_layer("ConvCaps2D_3", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=32, inchannels=32, incapsules=4, kernsize=3, outsize=32, outchannels=32, outcapsules=4, period=self.period))

        self.add_layer("ConvCaps2D_4", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=32, inchannels=32, incapsules=4, kernsize=3, outsize=32, outchannels=32, outcapsules=4, period=self.period))

        self.add_layer("ConvCaps2D_5", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=32, inchannels=32, incapsules=4, kernsize=3, outsize=16, outchannels=32, outcapsules=8, period=self.period))

        self.add_layer("ConvCaps2D_6", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=16, inchannels=32, incapsules=8, kernsize=3, outsize=16, outchannels=32, outcapsules=8, period=self.period))

        self.add_layer("ConvCaps2D_7", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=16, inchannels=32, incapsules=8, kernsize=3, outsize=16, outchannels=32, outcapsules=8, period=self.period))

        self.add_layer("ConvCaps2D_8", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=16, inchannels=32, incapsules=8, kernsize=3, outsize=16, outchannels=32, outcapsules=8, period=self.period))

        self.add_layer("ConvCaps2D_9", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=16, inchannels=32, incapsules=8, kernsize=3, outsize=8, outchannels=32, outcapsules=8, period=self.period))

        self.add_layer("ConvCaps2D_10", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=8, inchannels=32, incapsules=8, kernsize=3, outsize=8, outchannels=32, outcapsules=8, period=self.period))

        self.add_layer("ConvCaps2D_11", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=8, inchannels=32, incapsules=8, kernsize=3, outsize=8, outchannels=32, outcapsules=8, period=self.period))

        self.add_layer("ConvCaps2D_12", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=8, inchannels=32, incapsules=8, kernsize=3, outsize=8, outchannels=32, outcapsules=8, period=self.period))

        self.add_layer("ConvCaps2D_13", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=8, inchannels=32, incapsules=8, kernsize=3, outsize=4, outchannels=32, outcapsules=8, period=self.period))

        self.add_layer("ConvCaps2D_14", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=4, inchannels=32, incapsules=8, kernsize=3, outsize=4, outchannels=32, outcapsules=8, period=self.period))

        self.add_layer("ConvCaps2D_15", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=4, inchannels=32, incapsules=8, kernsize=3, outsize=4, outchannels=32, outcapsules=8, period=self.period))


        self.add_layer("ConvCaps3D_conv", hwacc.ConvCaps3D(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,
                insize=4, inchannels=32, incapsules=8, kernsize=3, outsize=4, outchannels=32, outcapsules=8, period=self.period))

        for i in range(3):
            self.add_layer("ConvCaps3D_sum_%d" % i, hwacc.SumLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=4, inchannels=32, incapsules=8, kernsize=4, outsize=1, outchannels=10, outcapsules=16, period=self.period))
            self.add_layer("ConvCaps3D_update_%d" % i, hwacc.SumLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages,insize=4, inchannels=32, incapsules=8, kernsize=4, outsize=1, outchannels=10, outcapsules=16, period=self.period))


        self.add_layer("ClassCaps_class", hwacc.ClassLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=4, inchannels=32, incapsules=8, kernsize=4, outsize=1,
                 outchannels=32, outcapsules=8, period=self.period))

        #for i in range(3):
        #    self.add_layer("ClassCaps_sum_%d" % i, hwacc.SumLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, period=self.period))
        #    self.add_layer("ClassCaps_update_%d" % i, hwacc.SumLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, period=self.period))

    def force_capsnet(self):
        c1 = e.add_layer("conv1", hwacc.ConvLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=28,
                                               inchannels=1, incapsules=1, kernsize=9, outsize=20,
                                               outchannels=256, outcapsules=1, period=self.period))
        c2 = e.add_layer("primary", hwacc.PrimaryLayer(self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=20,
                                                        inchannels=256, incapsules=1, kernsize=9, outsize=6, outchannels=32,
                                                        outcapsules=8, period=self.period))
        c3 = e.add_layer("class", hwacc.ClassLayer(
            self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=6, inchannels=32, incapsules=8, kernsize=6, outsize=1,
                 outchannels=32, outcapsules=8, period=self.period))

        for i in range(5):
            e.add_layer("sum_%d" % i, hwacc.SumLayer(
                self.sa_rows, self.sa_cols, self.pe_width, self.pe_stages, insize=6, inchannels=32, incapsules=8, kernsize=6, outsize=1, outchannels=10, outcapsules=16, period=self.period))
                
if __name__ == "__main__":
    import json
    for fn in sys.argv[1:]:
        e = hw_estimator()
        if fn == "deepcaps":
            e.force_deepcaps()               
        elif fn == "capsnet":
            e.force_capsnet()
        else:
            print("#parsing ", fn)
            e.parse_genotype(json.load(open(fn)))    

        print(e)
        print("energy = ", e.get_energy())
        print("latency= ", e.get_latency(), " ns")
        print("memory = ", e.get_memory())
