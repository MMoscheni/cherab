import numpy as np
import math
from raysect.core.math import translate
from raysect.primitive import Cylinder

#from cherab.solps import load_solps_from_mdsplus
import pickle

from raysect.core import Point2D, Point3D

# auto build and import cython code
import pyximport
pyximport.install(inplace=True)

# for InhomogeneousVolumeEmitter in CoreEmitter, see:
# https://raysect.github.io/documentation/api_reference/optical/optical_volumes.html?highlight=inhomogeneousvolumeemitter
# with this CLASS it is possible to user-define the integration technique
#from core_noC import CoreEmitter
from import_SOLPS_rad import load_solps_from_file

# MMM
# from cherab.solps.models.solps_emitter import make_solps_emitter
from solps_discrete_emitter import make_solps_discrete_emitter

######################################################################################################################
######################################################################################################################
######################################################################################################################


# (2) load_sol_radiation: it is a functions that accepts as input the SOLPS parameter to get access to
#     the data, the core_radiation_mask that is the function exploited to create the 2D piecewise emission
#     function, the characteristics of the plasma;
#     it then returns the cylinder made of a customized (in step integration) material generated by the 
#     radiation function and the radiation function itself (2D interpolation)
def load_discrete_sol_radiation(config, parent = None):

    # obtain SOLPS simulation
    print('\nReading SOL data from FILE...\n')

    if parent is None:
      raise TypeError('A parent node, e.g. World(), must be provided!')

    run = config['run']
    input_dir = config['input_directory']
    SOLPSdataFile = config['plasma']['SOLPS']['SOLPS_data_file'] + run
    SOLPSdataFile = input_dir + "/" + run + "/" + SOLPSdataFile + ".mat"

    speciesList = config['plasma']['SOLPS']['SOLPS_species_list']
    typeRad = config['plasma']['SOLPS']['SOLPS_type_rad']

    ################

    solps_simulation = load_solps_from_file(SOLPSdataFile, speciesList, typeRad)
      
    integration_step = config['raytracing']['integration_step']

    coreEmitter = make_solps_discrete_emitter(solps_simulation.mesh,
                                              solps_simulation.total_radiation_f2d,
                                              parent = parent,
                                              step = integration_step,
                                              configFile = config)

    return coreEmitter

