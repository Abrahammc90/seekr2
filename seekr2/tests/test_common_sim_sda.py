"""
test_sim_sda.py
"""

import pytest
import os
import re
from shutil import copyfile

import parmed
import numpy as np

import seekr2.modules.common_sim_sda as sim_sda

TEST_DIRECTORY = os.path.dirname(__file__)

def make_test_sda_input(tmp_path):
    test_filename = os.path.join(tmp_path, "test_input.in")
    sda_input = sim_sda.Input()
    sda_input.nrun = 1000
    sda_input.timemax = 0
    sda_input.dseed = 150
    sda_input.type = "sda_2proteins"
    sda_input.solutes = []
    sda_input.ftrajectories = "test_traj"
    sda_input.fcomplexes = "test_complexes"

    return test_filename, sda_input

def make_test_apbs_input(tmp_path):

    apbs = sim_sda.APBS_Input()
    test_filename = os.path.join(tmp_path, "apbs.in")
    ion1 = sim_sda.Ion()
    ion1.radius = 1.34
    ion1.charge = -2.0
    ion1.conc = 0.2
    ion2 = sim_sda.Ion()
    ion2.radius = 0.85
    ion2.charge = 1.0
    ion2.conc = 0.4
    apbs.solvent.ions = [ion1, ion2]
    apbs.dielectric = 77.4
    apbs.temperature = 298.15
    apbs.relative_viscosity = 0.4
    apbs.kT = 1.456
    apbs.desolvation_parameter = 1.0

    return test_filename, apbs

def make_test_sda_grids(tmp_path):

    sda_grid_inputs = sim_sda.SDA_grid_inputs()
    ep_test_filename = os.path.join(tmp_path, "ep.in")
    ed_test_filename = os.path.join(tmp_path, "ed.in")
    ed_test_filename = os.path.join(tmp_path, "hd.in")
    #lj_test_filename = os.path.join(tmp_path, "lj.in")

    ion1 = sim_sda.Ion()
    ion1.radius = 1.34
    ion1.charge = -2.0
    ion1.conc = 0.2
    ion2 = sim_sda.Ion()
    ion2.radius = 0.85
    ion2.charge = 1.0
    ion2.conc = 0.4

    sda_grid_inputs.temperature = 298.15

    ionic_strength = 0
    ionic_strength += ion1.conc*ion1.charge**2 / 2
    ionic_strength += ion2.conc*ion2.charge**2 / 2
    sda_grid_inputs.ionic_strength = ionic_strength


    return ep_test_filename, ed_test_filename, ed_test_filename, \
           sda_grid_inputs


def test_create_ghost_atom_from_atoms_center_of_mass(tmp_path):
    input_pqr_filename = \
        os.path.join(os.path.dirname(__file__), 
                     "../data/hostguest_files/hostguest_ligand.pqr")
    atom_index_list = list(range(15))
    output_pqr_filename = os.path.join(tmp_path, "test.pqr")
    ghost_atom = sim_sda.create_ghost_atom_from_atoms_center_of_mass(
        input_pqr_filename, atom_index_list, output_pqr_filename)
    pqr_struct = parmed.load_file(output_pqr_filename, skip_bonds=True)
    ghost_index = len(pqr_struct.atoms)
    #ghost_atom = pqr_struct.atoms[ghost_index-1]
    ghost_atomname = ghost_atom[12:16].strip()
    assert(ghost_atomname == "GHO")
    expected_ghost_location = np.array([[0.0, 0.0, 0.0]])
    ghost_location = pqr_struct.coordinates[ghost_index-1]
    difference = np.linalg.norm(expected_ghost_location - ghost_location)
    assert(difference == 0.0)
    return