
# Copyright 2016-2021 Euratom
# Copyright 2016-2021 United Kingdom Atomic Energy Authority
# Copyright 2016-2021 Centro de Investigaciones Energéticas, Medioambientales y Tecnológicas
#
# Licensed under the EUPL, Version 1.1 or – as soon they will be approved by the
# European Commission - subsequent versions of the EUPL (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#
# https://joinup.ec.europa.eu/software/page/eupl5
#
# Unless required by applicable law or agreed to in writing, software distributed
# under the Licence is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
#
# See the Licence for the specific language governing permissions and limitations
# under the Licence.

from raysect.core import translate
from raysect.optical.material import VolumeTransform
from raysect.primitive import Cylinder, Subtract

from cherab.core.math.mappers import AxisymmetricMapper
from cherab.tools.emitters import RadiationFunction

from cherab.solps import SOLPSMesh, SOLPSFunction2D

# MMM
import numpy as np                
from raysect.core import Point3D, rotate_z
from raysect.primitive import Box

def make_solps_interval_emitter(solps_mesh, radiation_function, parent=None, step=0.01): # MMM
    """
    Non-spectral emitter with the emissivity defined as SOLPSFunction2D.

    :param SOLPSMesh solps_mesh: SOLPS simulation mesh.
    :param SOLPSFunction2D radiation_function: Emissivity in W m-3.
    :param Node parent: parent node in the scenegraph, e.g. a World object.
    :param float step: Volume integration step in meters.

    :rtype: Primitive
    """

    if not isinstance(solps_mesh, SOLPSMesh):
        raise TypeError('Argument solps_mesh must be a SOLPSMesh instance.')
    if not isinstance(radiation_function, SOLPSFunction2D):
        raise TypeError('Argument radiation_function must be a SOLPSFunction2D instance.')

    radiation_function_3d = AxisymmetricMapper(radiation_function)
    outer_radius = solps_mesh.mesh_extent['maxr']
    inner_radius = solps_mesh.mesh_extent['minr']
    height = solps_mesh.mesh_extent['maxz'] - solps_mesh.mesh_extent['minz']
    lower_z = solps_mesh.mesh_extent['minz']
    emitter = RadiationFunction(radiation_function_3d, step = step)
    material = VolumeTransform(emitter, transform = translate(0, 0, -lower_z))

    # conventional toroidal plasma emitter # MMM
    # plasma_volume = Subtract(Cylinder(outer_radius, height), Cylinder(inner_radius, height),
    #                         material=material, parent=parent, transform=translate(0, 0, lower_z))

    # MMM whatever comes next
    #
    # CHERAB sees an emitter if World contains at least one
    # (material + enclosing primitive), i.e. whatever outside
    # a primitive boundary is NOT seen as a source.
    #
    # Leading idea:
    #
    # - create a toroidally symmetric source of radiation
    # - create discrete slices of the usual hollow cylinder
    #   (toroidally symmetric object)
    # 
    # => toroidally symmetric source 
    #    +
    #    discrete primitives
    #    =
    #    discrete source 

    # create complete hollow cylinder
    hollow_cylinder = Subtract(Cylinder(outer_radius, height),
                               Cylinder(inner_radius, height))
    
    # create boxes (need to import them) to slice hollow cylinder:
    # - one side through the origin 
    # - width  > hollow_cylinder diameter
    # - height > hollow_cylinder height
    # - translated so as mid-height at R-phi plane (Z=0)

    # - emission exist only on limiters (i.e. where plasma touches walls)
    # - angular_width_sector = angular_width_lim + angular_width_nonlim
    #   and
    #   angular_width_sector_deg = 360 / num_limiters

    num_limiters = 8                                                  # [ - ] 8 limiters = 8 non-limiters
    angular_width_lim = 10                                            # [deg] width of the limiters ????
    angular_width_sector_deg = 360 / num_limiters                     # [deg]
    angular_width_sector_rad = angular_width_sector_deg * np.pi / 180 # [rad] 
    padding = 1E-01                                                   # [m]   some safety margin

    box_height = 2 * (np.max(np.abs([min_z, max_z])) + padding)
    box_width  = 2 * (2 * outer_radius + padding)

    # BOX 1:
    # one side through the origin along x axis

    lower_point = Point3D(- 0.5 * box_width,
                          - 0.5 * box_width,
                          - 0.5 * box_height)
    upper_point = Point3D(+ 0.5 * box_width,
                          + 0.0,
                          + 0.5 * box_height)

    box_1 = Box(lower_point, upper_point)

    # BOX 2:
    # one side through the origin along axis rotated by 360/8 degrees
    #
    # CAUTION. 
    # rotating_z and translating box 1 may be ok, but does rotate_z 
    # does as such along Z axis of World or of box_1???
    # doing by hand for the time being...

    lower_point = Point3D(- 0.5 * box_width * np.cos(angular_width_sector_rad),
                          - 0.5 * box_width * np.sin(angular_width_sector_rad),
                          - 0.5 * box_height)
    upper_point = Point3D(+ 0.5 * box_width * np.cos(angular_width_sector_rad),
                          + 0.5 * box_width * np.sin(angular_width_sector_rad),
                          + 0.5 * box_height)

    box_2 = Box(lower_point, upper_point)
    
    # subtract to create slice from original complete hollow cylinder
    # must be suitably rotated_z to align with camera

    plasma_volume = Subtract(hollow_cylinder, box_1)
    plasma_volume = Subtract(plasma_volume,   box_2)
    
    # CAUTION:
    # this only 1 slice, you need several (ask Chris how many) of them
    # that you can obtain by rotating this one around the torus
    # (NO NEED TO RE-BUILD FROM SCRATCH)
    #
    # i.e. similarly to the well-known:
    # mesh.instance(parent = world, material = AbsorbingSurface(), transform = rotate(0, 0, i * periodicity))
    #
    # should work, both a mesh and a volume are Raysect Primitives...
    # see https://www.raysect.org/api_reference/core/raysect_core_scenegraph.html?highlight=instance#raysect.core.scenegraph.primitive.Primitive.instance
    # and look for "instance"

    for i in range(num_limiters):
        plasma_volume.instance(parent = world,
                               material = material,
                               transform = rotate_z(i * angular_width_sector_deg))

    return plasma_volume
