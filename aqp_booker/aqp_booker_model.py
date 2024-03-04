from dataclasses import dataclass

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class AqpBookerParameters:
    """
    Contains the default parameters for QT3 atomic quantum processor magnet geometry
    Base units for magpylib are millimeters, milliteslas, degrees, and Amperes
    """

    # assume x, y, z coils are all round, may consider racetrack coil later
    # assume x and y axis coils are symmetric with respect to the sample center, current has same magnitude, in the opposite direction
    # assume z axis coil are not symmetric

    # shape of the coils, can be 'round' or 'racetrack'
    # race track means the coil is a rectangle with rounded corners
    # x, y and z can be 0, which means the coil is round with r "radius"
    #    
    #              x/y/z
    #         ___________  radius
    #        ◜           ◝
    #        |            |
    #        |            |
    #  x/y/z |            |
    #        |            |
    #        |            |
    #        |            |
    #        ◟___________◞
    # 


    x_axis_coil_y_length: float = 0
    x_axis_coil_z_length: float = 60
    x_axis_coil_corner_radius: float = 18
    x_axis_coil_winding_number: int = 200
    x_axis_coil_current_a: float = -1.5
    x_axis_coil_current_b: float = 1.5
    x_axis_coil_center_a: tuple = (38, 0, 0)
    x_axis_coil_center_b: tuple = (-38, 0, 0)


    y_axis_coil_x_length: float = 0
    y_axis_coil_z_length: float = 60
    y_axis_coil_corner_radius: float = 18
    y_axis_coil_winding_number: int = 200
    y_axis_coil_current_a: float = -1.5
    y_axis_coil_current_b: float = 1.5
    y_axis_coil_center_a: tuple = (0, 38, 0)
    y_axis_coil_center_b: tuple = (0, -38, 0)

    # z axis coil not used for 2D MOT
    z_axis_coil_x_length: float = 27
    z_axis_coil_y_length: float = 27
    z_axis_coil_corner_radius: float = 10
    z_axis_coil_winding_number: int = 1
    z_axis_coil_current_a: float = -1*0
    z_axis_coil_current_b: float = 0
    z_axis_coil_center_a: tuple = (0, 0, 80)
    z_axis_coil_center_b: tuple = (0, 0, -80)

    sample_center: tuple = (0, 0, 0)

    x_wire_length: float = ((x_axis_coil_y_length + x_axis_coil_z_length)*2 + x_axis_coil_corner_radius*2*np.pi) * x_axis_coil_winding_number /1000 # meters
    y_wire_length: float = ((y_axis_coil_x_length + y_axis_coil_z_length)*2 + y_axis_coil_corner_radius*2*np.pi) * y_axis_coil_winding_number /1000 # meters
    z_wire_length: float = ((z_axis_coil_x_length + z_axis_coil_y_length)*2 + z_axis_coil_corner_radius*2*np.pi) * z_axis_coil_winding_number /1000 # meters

    wire_selection: int = 24 # AWG
    wire_diameter: float = 0.127 * (92 ** ((36 - wire_selection)/39)) # mm
    # R = rho * L / A
    x_wire_resistance: float = (1.7241*10**(-8)) * x_wire_length / (np.pi*((wire_diameter/2/1000) ** 2))  # ohm
    y_wire_resistance: float = (1.7241*10**(-8)) * y_wire_length / (np.pi*((wire_diameter/2/1000) ** 2))  # ohm
    z_wire_resistance: float = (1.7241*10**(-8)) * z_wire_length / (np.pi*((wire_diameter/2/1000) ** 2))  # ohm

    wire_weight: float = x_wire_length * np.pi * ((wire_diameter/2000) ** 2) * 8941

    # winding_layer_number: int = 3
    # x_axis_coil_thickness: float = x_axis_coil_winding_number * wire_diameter / winding_layer_number
    # y_axis_coil_thickness: float = y_axis_coil_winding_number * wire_diameter / winding_layer_number
    # z_axis_coil_thickness: float = z_axis_coil_winding_number * wire_diameter / winding_layer_number

    # using fixed thickness
    x_axis_coil_thickness: float = 20
    y_axis_coil_thickness: float = 20
    z_axis_coil_thickness: float = 20

    position_tolorance: float = 1 # mm
    angle_tolorance: float = 0.1 # degree

def get_default_aqp_parameters():
    p = AqpBookerParameters()
    return p

def build_magpy_collection(p: AqpBookerParameters = get_default_aqp_parameters()) -> magpy.Collection:

    # assume the coils are all rounds.
    x_axis_coll = magpy.Collection()
    y_axis_coll = magpy.Collection()
    z_axis_coll = magpy.Collection()
    # field_sensor = magpy.Sensor(position=p.sample_center, style_label='field_sensor')
    # coll = magpy.Collection(x_axis_coll, y_axis_coll, z_axis_coll, field_sensor)
    coll = magpy.Collection(x_axis_coll, y_axis_coll, z_axis_coll)

    # sample_number = 1000
    # ts = np.linspace(-1 * p.x_axis_coil_winding_number/2, p.x_axis_coil_winding_number/2, sample_number)
    x_vertices = shape_racetrack(p.x_axis_coil_winding_number, p.x_axis_coil_corner_radius, p.x_axis_coil_y_length, p.x_axis_coil_z_length, p.x_axis_coil_thickness) 

    x_axis_coil_a = magpy.current.Line(current=p.x_axis_coil_current_a, vertices=x_vertices, position=p.x_axis_coil_center_a, style_label='x_axis_coil_a', style_color = 'r')
    x_axis_coil_b = magpy.current.Line(current=p.x_axis_coil_current_b, vertices=x_vertices, position=p.x_axis_coil_center_b, style_label='x_axis_coil_b', style_color = 'g')

    x_axis_coil_a.rotate_from_angax(90, 'y')
    x_axis_coil_b.rotate_from_angax(90, 'y')
    x_axis_coil_a.rotate_from_angax(90, 'x')
    x_axis_coil_b.rotate_from_angax(90, 'x')
    x_axis_coll.add(x_axis_coil_a, x_axis_coil_b)

    # y_axis_coil_a = magpy.current.Loop(current = p.y_axis_coil_current_a, diameter=p.y_axis_coil_diameter_a,\
    #                                     position= p.y_axis_coil_center_a, style_label='y_axis_coil_a')
    # y_axis_coil_b = magpy.current.Loop(current = p.y_axis_coil_current_b, diameter=p.y_axis_coil_diameter_b,\
    #                                     position= p.y_axis_coil_center_b, style_label='y_axis_coil_b')

    y_vertices = shape_racetrack(p.y_axis_coil_winding_number, p.y_axis_coil_corner_radius, p.y_axis_coil_x_length, p.y_axis_coil_z_length, p.y_axis_coil_thickness)
    y_axis_coil_a = magpy.current.Line(current=p.y_axis_coil_current_a, vertices=y_vertices, position=p.y_axis_coil_center_a, style_label='y_axis_coil_a', style_color='b')
    y_axis_coil_b = magpy.current.Line(current=p.y_axis_coil_current_b, vertices=y_vertices, position=p.y_axis_coil_center_b, style_label='y_axis_coil_b', style_color='y')
    y_axis_coil_a.rotate_from_angax(90, 'x')
    y_axis_coil_b.rotate_from_angax(90, 'x')
    y_axis_coll.add(y_axis_coil_a, y_axis_coil_b)

    # z_axis_coil_a = magpy.current.Loop(current = p.z_axis_coil_current_a, diameter=p.z_axis_coil_diameter_a,\
    #                                     position= p.z_axis_coil_center_a, style_label='z_axis_coil_a')
    # z_axis_coil_b = magpy.current.Loop(current = p.z_axis_coil_current_b, diameter=p.z_axis_coil_diameter_b,\
    #                                     position= p.z_axis_coil_center_b, style_label='z_axis_coil_b')

    z_vertices = shape_racetrack(p.z_axis_coil_winding_number, p.z_axis_coil_corner_radius, p.z_axis_coil_x_length, p.z_axis_coil_y_length, p.z_axis_coil_thickness)
    z_axis_coil_a = magpy.current.Line(current=p.z_axis_coil_current_a, vertices=z_vertices, position=p.z_axis_coil_center_a, style_label='z_axis_coil_a', style_color='m')
    z_axis_coil_b = magpy.current.Line(current=p.z_axis_coil_current_b, vertices=z_vertices, position=p.z_axis_coil_center_b, style_label='z_axis_coil_b', style_color='c')
    z_axis_coil_a.rotate_from_angax(90, 'z')
    z_axis_coil_b.rotate_from_angax(90, 'z')
    z_axis_coll.add(z_axis_coil_a, z_axis_coil_b)

    return coll

def simulate_mag_field(p: AqpBookerParameters = get_default_aqp_parameters()):
    coll = build_magpy_collection(p)
    # field_sensor = coll.sensors[0]
    print("x direction gradient is", get_gradient(coll, np.asarray(p.sample_center), 'x'))
    print("y direction gradient is", get_gradient(coll, np.asarray(p.sample_center), 'y'))
    print("z direction gradient is", get_gradient(coll, np.asarray(p.sample_center), 'z'))
    image = magpy.show(coll)

def shape_racetrack(winding_number, corner_radius, edge_1_length, edge_2_length, coil_thickness) -> np.ndarray:
    sample_number = 10000
    ts = np.linspace(-1 * winding_number/2, winding_number/2, sample_number)
    thick = np.linspace(-1 * coil_thickness/2, coil_thickness/2, sample_number)
    vertices = np.c_[corner_radius*np.cos(ts*2*np.pi), corner_radius*np.sin(ts*2*np.pi), thick]
    for i in range(len(vertices)):
        if (np.cos(ts[i] * 2 * np.pi) >= 0 and np.sin(ts[i] * 2 * np.pi) >= 0):
            vertices[i][0] += edge_1_length/2
            vertices[i][1] += edge_2_length/2
        elif (np.cos(ts[i] * 2 * np.pi) < 0 and np.sin(ts[i] * 2 * np.pi) >= 0):
            vertices[i][0] -= edge_1_length/2
            vertices[i][1] += edge_2_length/2
        elif (np.cos(ts[i] * 2 * np.pi) < 0 and np.sin(ts[i] * 2 * np.pi) < 0):
            vertices[i][0] -= edge_1_length/2
            vertices[i][1] -= edge_2_length/2
        elif (np.cos(ts[i] * 2 * np.pi) >= 0 and np.sin(ts[i] * 2 * np.pi) < 0):
            vertices[i][0] += edge_1_length/2
            vertices[i][1] -= edge_2_length/2

    vertices = np.append(vertices, [[vertices[0][0], vertices[0][1], vertices[-1][2]]], axis=0)
    # print(vertices)
    return vertices


def get_gradient(coll: magpy.Collection, location: np.ndarray, direction: str) -> float:
    
    diff = 0.05 # mm
    B0 = coll.getB(location)
    if direction == 'x':
        B1 = coll.getB(location + np.array([diff, 0, 0]))
        gradient = (B1[0] - B0[0])/diff # mT/mm
    elif direction == 'y':
        B1 = coll.getB(location + np.array([0, diff, 0]))
        gradient = (B1[1] - B0[1])/diff
    elif direction == 'z':
        B1 = coll.getB(location + np.array([0, 0, diff]))
        gradient = (B1[2] - B0[2])/diff
    else:
        raise ValueError("direction must be one of 'x', 'y', 'z'")
    
    gradient = gradient * 100 # convert to G/cm
    if B0 is None or B1 is None:
        raise ValueError("B0 or B1 is None")

    return gradient

def heat_generation(p: AqpBookerParameters = get_default_aqp_parameters()):
    # heat map for winding number and current
    winding_min = 50
    winding_max = 300
    current_min = 1
    current_max = 3
    sample_points = 50
    winding_number = np.linspace(winding_min, winding_max, sample_points)
    current = np.linspace(current_min, current_max, sample_points)
    data = np.zeros((len(winding_number), len(current)))
    data_heat = np.zeros((len(winding_number), len(current)))
    for i in range(len(winding_number)):
        for j in range(len(current)):
            print("calculating: ", i, ", ", j)
            p.x_axis_coil_winding_number = winding_number[i]
            p.y_axis_coil_winding_number = winding_number[i]
            p.x_axis_coil_current_a = -1 * current[j]
            p.x_axis_coil_current_b = current[j]
            p.y_axis_coil_current_a = -1 * current[j]
            p.y_axis_coil_current_b = current[j]
            coll = build_magpy_collection(p)
            data[i][j] = get_gradient(coll, np.asarray(p.sample_center), 'x')
            if np.abs(data[i][j]) > 15:
                x_heat_production = p.x_wire_resistance * p.x_axis_coil_current_a**2
                wire_weight = p.wire_weight
                delta_temp = x_heat_production * 1 / 0.385 / (wire_weight * 1000)
                data_heat[i][j] = delta_temp
    # export the data to csv
    
    np.savetxt("data.csv", data, delimiter=",")
    np.savetxt("data_heat.csv", data_heat, delimiter=",")
    plt.subplot(1, 2, 1)
    plt.imshow(data, cmap='hot', interpolation='nearest', extent=[current_min, current_max, winding_max, winding_min], aspect='auto')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(data_heat, cmap='hot', interpolation='nearest', extent=[current_min, current_max, winding_max, winding_min], aspect='auto')
    plt.colorbar()
    plt.show()


    
def B_field_plot(p: AqpBookerParameters = get_default_aqp_parameters()):
    coll = build_magpy_collection(p)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    # plot along x axis
    x_axis_field = []
    for i in np.linspace(-10 * p.z_axis_coil_x_length/2, 10*p.z_axis_coil_x_length/2, 100):
        B = coll.getB((i, p.x_axis_coil_center_a[1], p.x_axis_coil_center_a[2]))
        x_axis_field.append(np.abs(B[0]))
    ax1.plot(np.linspace(-1 * p.z_axis_coil_x_length/2, p.z_axis_coil_x_length/2, 100), x_axis_field)
    ax1.set_xlabel('x (mm)')
    ax1.set_ylabel('B (mT)')
    ax1.set_title('B field along x axis')

    # plot along y axis
    y_axis_field = []
    for i in np.linspace(-10 * p.z_axis_coil_y_length/2, 10* p.z_axis_coil_y_length/2, 100):
        B = coll.getB((p.y_axis_coil_center_a[0], i, p.y_axis_coil_center_a[2]))
        y_axis_field.append(np.abs(B[1]))
    ax2.plot(np.linspace(-1 * p.z_axis_coil_y_length/2, p.z_axis_coil_y_length/2, 100), y_axis_field)
    ax2.set_xlabel('y (mm)')
    ax2.set_ylabel('B (mT)')
    ax2.set_title('B field along y axis')

    # plot along z axis
    z_axis_field = []
    for i in np.linspace(-1 * p.x_axis_coil_z_length/2, p.x_axis_coil_z_length/2, 100):
        B = coll.getB((p.z_axis_coil_center_a[0], p.z_axis_coil_center_a[1], i))
        z_axis_field.append(np.abs(B[2]))
    ax3.plot(np.linspace(-1 * p.x_axis_coil_z_length/2, p.x_axis_coil_z_length/2, 100), z_axis_field)
    ax3.set_xlabel('z (mm)')
    ax3.set_ylabel('B (mT)')
    ax3.set_title('B field along z axis')

    plt.tight_layout()
    plt.show()


def resistance_sim(p:AqpBookerParameters = get_default_aqp_parameters()):
    print("coil diameter is", p.wire_diameter, "mm")
    print("coil thickness is", p.x_axis_coil_thickness, "mm")
    print("x coil resistance is", p.x_wire_resistance, "ohm")
    print("y coil resistance is", p.y_wire_resistance, "ohm")
    print("z coil resistance is", p.z_wire_resistance, "ohm")
    print("x coil wire length is", p.x_wire_length, "m")

def heat_sim(p:AqpBookerParameters = get_default_aqp_parameters()):
    x_heat_production = p.x_wire_resistance * p.x_axis_coil_current_a**2
    print("Each x coil heat production is", p.x_wire_resistance * p.x_axis_coil_current_a**2, "W")
    print("Each y coil heat production is", p.y_wire_resistance * p.y_axis_coil_current_a**2, "W")
    # print("z coil heat production is", p.z_wire_resistance * p.z_axis_coil_current_a**2, "W")
    wire_weight = p.x_wire_length * np.pi * ((p.wire_diameter/2000) ** 2) * 8941 # kg
    print("x coil wire weight is", wire_weight, "kg")
    delta_temp = x_heat_production * 1 / 0.385 / (wire_weight * 1000) # assume only 1 second on time
    print("x coil temperature rise is", delta_temp, "K")

def find_lowest_heat_gen(p:AqpBookerParameters = get_default_aqp_parameters()):
    winding_min = 50
    winding_max = 300
    current_min = 0.5
    current_max = 3.0
    sample_points = 50

    gradient_limit = 18 # G/cm

    winding_number = np.linspace(winding_min, winding_max, sample_points, dtype=int)
    current = np.linspace(current_min, current_max, sample_points)

    gradient_list = []

    for i in range(len(winding_number)):
        for j in range(len(current)):
            print("calculating: ", winding_number[i], " windings, ", current[j], " A\n")
            p.x_axis_coil_winding_number = winding_number[i]
            p.y_axis_coil_winding_number = winding_number[i]
            p.x_axis_coil_current_a = -1 * current[j]
            p.x_axis_coil_current_b = current[j]
            p.y_axis_coil_current_a = -1 * current[j]
            p.y_axis_coil_current_b = current[j]
            coll = build_magpy_collection(p)
            gradient = get_gradient(coll, np.asarray(p.sample_center), 'x')
            if np.abs(gradient) > gradient_limit:
                x_heat_production = p.x_wire_resistance * p.x_axis_coil_current_a**2
                
                gradient_list.append([winding_number[i], current[j], gradient, x_heat_production])
                break
    
    # export the data to csv
    np.savetxt("gradient_list_24awg.csv", gradient_list, delimiter=",")
    print("done")

if __name__ == "__main__":
    resistance_sim()
    heat_sim()
    simulate_mag_field()
    # heat_generation()
    B_field_plot()
    # find_lowest_heat_gen()