from dataclasses import dataclass

import magpylib as magpy
import matplotlib.pyplot as plt
import numpy as np

"""
To-do:
1. Refer the initial position of the adjustable magnet from the QdpWillowParameters object instead of re-writing
the initial position within show_move_sample_along_y()

"""


@dataclass
class QdpWillowParameters:
    # Base units for magpylib are mm and mT
    cryo_magnet_diameter: float
    cryo_magnet_height: float
    cryo_magnet_center_xyz: np.ndarray
    cry_magnet_remanence: np.ndarray

    ext_magnet_diameter: float
    ext_magnet_height: float
    ext_magnet_center_xyz: np.ndarray
    ext_magnet_remanence: np.ndarray

    sample_center: np.ndarray


def get_default_qdp_params() -> QdpWillowParameters:
    """

    Gets the default parameters for QT3s quantum diamond processor magnet geometry.
    :return: Parameters

    """
    p = QdpWillowParameters(cryo_magnet_diameter=15,
                            cryo_magnet_height=8,
                            cryo_magnet_center_xyz=np.array([-23, 0, -4+1.0668]),
                            cry_magnet_remanence=np.array([0, 0, 1050]),
                            ext_magnet_diameter=25.4,
                            ext_magnet_height=19.05,
                            ext_magnet_center_xyz=np.array([-64, 0, 0]),
                            ext_magnet_remanence=np.array([0, 0, 1480]),
                            sample_center=np.array([0, 0, 0]))
    return p


def build_magpy_collection(p: QdpWillowParameters) -> magpy.Collection:
    """

    Creates a collection of source and sensor objects based on the parameters settings and returns the collection.
    :param p: Parameters dataclass instance
    :return: magpylib collection

    """
    coll = magpy.Collection()

    cryo_magnet = magpy.magnet.Cylinder(magnetization=p.cry_magnet_remanence,
                                        dimension=(p.cryo_magnet_diameter, p.cryo_magnet_height),
                                        position=p.cryo_magnet_center_xyz,
                                        style_label='cryo_magnet')
    cryo_magnet.parent = coll

    ext_magnet = magpy.magnet.Cylinder(magnetization=p.ext_magnet_remanence,
                                       dimension=(p.ext_magnet_diameter, p.ext_magnet_height),
                                       position=p.ext_magnet_center_xyz,
                                       style_label='ext_magnet')
    ext_magnet.parent = coll

    sample = magpy.Sensor(position=p.sample_center, style_label='sample')
    sample.parent = coll

    return coll


def show_move_sample_along_y(p: QdpWillowParameters = get_default_qdp_params(), yrange=(-25, 25), num=100):
    """

    Moves sample (usually at origin by default) between the given values in x_range.
    Plots an animation of the motion relative to magnet as well as a range.
    :param p: Parameters dataclass
    :param yrange: (min_y, max_y) tuple for sample position range
    :param num: Number of points in sample position sweep
    :return: None, just shows plots.

    """
    coll = build_magpy_collection(p)

    adjustable_magnet = coll.sources[1]
    adjustable_magnet.rotate_from_angax(90, 'y')
    adjustable_magnet.position = np.linspace((-64, yrange[0], 0), (-64, yrange[1], 0), num=num)
    y_lin = np.linspace(yrange[0], yrange[1], num=num)

    sample = coll.sensors[0]
    B = sample.getB(coll) / 1000  # B-field in Tesla

    coll.show(backend='plotly', animation=2)
    fig, ax = plt.subplots(1, 1)
    ax.plot(y_lin, B * 1.E4, label=['Bx', 'By', 'Bz'])
    ax.set_xlabel('External Magnet y-position (mm)')
    ax.set_ylabel('Total B-Field (Gauss)')
    fig.legend()
    ax.grid()
    plt.show()


if __name__ == "__main__":
    show_move_sample_along_y()
