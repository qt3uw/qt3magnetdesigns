from dataclasses import dataclass

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np

sample_number = 1000

winding_number = 1
coil_radius = 100
coil_thickness = 1

current = 1


ts = np.linspace(-1 * winding_number/2, winding_number/2, sample_number)
thick = np.linspace(-1 * coil_thickness/2, coil_thickness/2, sample_number)
vertices = np.c_[coil_radius*np.cos(ts*2*np.pi), coil_radius*np.sin(ts*2*np.pi), thick]
vertices = np.append(vertices, [[vertices[0][0], vertices[0][1], vertices[-1][2]]], axis=0)

# coil = magpy.current.Line(current=current, vertices=vertices, position=(0,0,0), style_label='coil_1', style_color = 'r')

coil = magpy.current.Loop(current=current, diameter=2 * coil_radius, style_color='b')

sensor = magpy.Sensor(position=(0,0,0), style_label='sensor_1')
coll = magpy.Collection(coil, sensor)
B = coll.getB()
print(B)
magpy.show(coll)