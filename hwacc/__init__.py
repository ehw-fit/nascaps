
from . import generic_layer
from . import conv_layer
from . import class_layer
from . import caps_layer
from . import sum_update_layer
from . import capscell_layer
from . import primary_layer
from . import dcapscell_layer
from . import convcaps3d_layer

ConvLayer = conv_layer.ConvLayer
ClassLayer = class_layer.ClassLayer
SumUpdateLayer = sum_update_layer.sum_update_layer
CapsLayer = caps_layer.caps_layer
CapsCellLayer = capscell_layer.CapsCellLayer
DCapsCellLayer = dcapscell_layer.DCapsCellLayer
PrimaryLayer = primary_layer.PrimaryLayer
ConvCaps3D = convcaps3d_layer.convcaps3d_layer

# aliases
SumLayer = SumUpdateLayer
UpdateLayer = SumUpdateLayer
