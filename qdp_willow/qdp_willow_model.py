from dataclasses import dataclass

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class QdpWillowParameters:
    # All SI units
    cryo_magnet_diameter: float
    cryo_magnet_height: float
    cryo_magnet_center_xyz: np.ndarray
    cry_magnet_remanence: np.ndarray
    sample_center: np.ndarray
    sample_normal_axis: np.ndarray


def get_default_qdp_magnet_parameters() -> QdpWillowParameters:
    """
    Gets the default parameters for QT3s quantum diamond processor magnet geometry.
    :return: Parameters
    """
    p = QdpWillowParameters(cryo_magnet_diameter=19.05E-3,
                            cryo_magnet_height=19.05E-3,
                            cryo_magnet_center_xyz= np.array([-24.5E-3, 0, 0]),
                            cry_magnet_remanence=np.array([0, 0, 1]),
                            sample_center=np.array([0, 0, 0]),
                            sample_normal_axis=np.array([0, 0, 1]))
    return p


def build_magpy_collection(p: QdpWillowParameters)->magpy.Collection:
    """
    Creates a collection of source and sensor objects based on the parameters settings and returns the collection.
    :param p: Parameters dataclass instance
    :return: magpylib collection
    """
    dqp = magpy.Collection()

    cryo_magnet = magpy.magnet.Cylinder(magnetization=p.cry_magnet_remanence*1.E3,
                                        dimension=(p.cryo_magnet_diameter*1.E3, p.cryo_magnet_height*1.E3),
                                        position=p.cryo_magnet_center_xyz * 1.E3,
                                        style_label='cryo_magnet')
    cryo_magnet.parent = dqp

    sample = magpy.Sensor(position=p.sample_center, style_label='sample')
    sample.parent = dqp
    return dqp


def show_move_sample_along_x(p: QdpWillowParameters=get_default_qdp_magnet_parameters(), xrange=(-1.E-3, 1.E-3), num=100):
    """
    Moves sample (usually at origin by default) between the given values in x_range.
    Plots an animation of the motion relative to magnet as well as a range.
    :param p: Parameters dataclass
    :param xrange: (min_x, max_x) tuple for sample position range
    :param num: Number of points in sample position sweep
    :return: None, just shows plots.
    """
    dqp = build_magpy_collection(p)
    sample = dqp.sensors[0]
    sample.position = np.linspace((xrange[0]*1.E3, 0, 0), (xrange[1]*1.E3, 0, 0), num=num)
    x_lin = np.linspace(xrange[0], xrange[1], num=num)

    #sample.move(np.linspace((xrange[0], 0, 0), (xrange[1], 0, 0), num=100))
    B = sample.getB(dqp) / 1000 # B-field in tesla

    dqp.show(backend='plotly', animation=2)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x_lin * 1.E3, B * 1.E4, label=['Bx', 'By', 'Bz'])
    ax.set_xlabel('sample x (mm)')
    ax.set_ylabel('Field (gauss)')
    fig.legend()
    ax.grid()
    plt.show()


if __name__ == "__main__":
    show_move_sample_along_x()



