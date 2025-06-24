import numpy as np
from .units import Units
from ..interfaces.adfsuite.ams import AMSJob
from ..mol.molecule import Molecule
import os

__all__ = ['get_stoichiometry', 'balance_equation', 'reaction_energy']

def get_stoichiometry(job_or_molecule_or_path, as_dict=True):
    r = job_or_molecule_or_path
    d = None
    if isinstance(r, AMSJob):
        d = r.molecule.get_formula(as_dict=as_dict)
    elif isinstance(r, Molecule):
        d = r.get_formula(as_dict=as_dict)
    elif isinstance(r, dict):
        d = r.copy()
    elif isinstance(r, str):
        if os.path.isdir(r):
            d = AMSJob.load_external(r).molecule.get_formula(as_dict=as_dict)
        elif os.path.exists(r):
            try:
                d = Molecule(r).get_formula(as_dict=as_dict)
            except:
                d = AMSJob.load_external(r).molecule.get_formula(as_dict=as_dict)
        else:
            raise ValueError("The path {} does not exist.".format(r))
                
    else:
        raise TypeError("expected type AMSJob or dict but received {}".format(type(r)))

    return d

def balance_equation(reactants, products, normalization='r0', normalization_value=1.0):
    """
    Calculate stoichiometric coefficients 
    This only works if 
        number_of_chemical_elements == len(reactants)+len(products), OR
        number_of_chemical_elements == len(reactants)+len(products)-1

    Returns: a 2-tuple (coeffs_reactants, coeffs_products)
        coeffs_reactants is a list with length == len(reactants)
        coeffs_products is a list with length == len(products)

    reactants: a list of amsjobs, or a list of paths to ams.results folders or ams.rkf files or .xyz files, or a list of Molecules, or a list of stoichiometry dicts, or a list of Molecules

    products: a list of amsjobs, or a list of paths to ams.results folders or ams.rkf files, or a list of Molecules or .xyz files, or a list of stoichiometry dicts, or a list of Molecules

    normalization: 'r0' for the first reactant, 'r1' for the second reactant, etc.
        'p0' for the first product, 'p1' for the second product, etc.
        This normalizes the chemical equation such that the coefficient in front of the specified species is normalization_value

    normalization_value : float
        The coefficient to normalize to

    EXAMPLE:

        balance_equation(
            reactants=[
                {'N': 2, 'H': 8, 'Cr': 2, 'O': 7}
            ],
            products=[
                {'Cr': 2, 'O': 3},
                {'N': 2},
                {'H': 2, 'O': 1}
            ])

        returns
        ([1.0], [1.0, 1.0, 1.0, 4.0])


    """
    def get_stoichiometries_and_elements(list_of_jobs):
        stoich = []
        elements = set()
        for r in list_of_jobs:
            d = get_stoichiometry(r)
            stoich.append(d)
            for k in d:
                elements.add(k)
        return stoich, elements

    def get_normalization_index(normalization):
        if normalization.startswith('r'):
            normalization_index = int(normalization.split('r')[1])
            if normalization_index >= num_reactants:
                raise ValueError("Reactant index {} specified, but max value allowed is {}".format(normalization_index, num_reactants-1))
        elif normalization.startswith('p'):
            normalization_index = int(normalization.split('p')[1]) 
            if normalization_index >= num_products:
                raise ValueError("Product index {} specified, but max value allowed is {}".format(normalization_index, num_products-1))
            normalization_index += num_reactants
        else:
            raise ValueError("Unknown normalization: {}. Should be r0, r1, r2, ... (for reactants), p0, p1, p2 ... (for products)")

        return normalization_index

    if len(reactants) == 0:
        raise ValueError('The reactants list is empty.')
    if len(products) == 0:
        raise ValueError('The products list is empty.')

    stoich_r, elements_r = get_stoichiometries_and_elements(reactants)
    stoich_p, elements_p = get_stoichiometries_and_elements(products)
    elements = elements_r
    elements.update(elements_p) #hopefully not necessary
    elements = list(elements)
    num_reactants = len(stoich_r)
    num_products = len(stoich_p)

    # EXAMPLE:
    # aCH4 + bO2 --> cCO2 + dH2O
    # mat =
    # [[-1 0    1 0], #C
    #  [-4 0    0 2], #H
    #  [0 -2    2 1]] #O
    #  CH4 O2 CO2 H2O
    #
    # Al2(SO4)3 + Ca(OH)2 → Al(OH)3 + CaSO4 
    # mat = np.array([[-2,-3,-12,0,0,0],[0,0,0,-1,-2,-2],[1,0,0,0,3,3],[0,1,4,1,0,0]]).T

    mat = []
    for e in elements:
        row = []
        for s in stoich_r:
            row.append(-s.get(e,0))
        for s in stoich_p:
            row.append(s.get(e,0))
        mat.append(row)
    mat = np.array(mat)

    coeffs = []
    if mat.shape[0] >= mat.shape[1]:
        u, s, vh = np.linalg.svd(mat)
        coeffs = np.compress(s <= 1e-12, vh, axis=0)
    elif mat.shape[0] == mat.shape[1]-1:
        # e.g. 3x4 matrix, so add a [1,1,1,1] row to find a particular solution
        newmat = np.concatenate((mat, np.array([[1]*mat.shape[1]])), axis=0)
        b = np.array([0]*(newmat.shape[0]-1)+[1.])
        try:
            coeffs = np.linalg.solve(newmat, b)
        except Exception as e:
            raise RuntimeError("Something went wrong when solving the system of linear equations. Verify that the chemical equation can be balanced at all, and that it can be balanced uniquely except for multiplication by a constant. {}\nA={}\nb={}".format(e, newmat,b))
    else:
        raise ValueError("The number of chemical elements must equal the number of molecules, or (the number of molecules-1). You have {} chemical elements: {}, and {} molecules".format(len(elements), elements, num_reactants+num_products))

    coeffs = coeffs.ravel()
    if len(coeffs) != num_reactants+num_products:
        raise RuntimeError("Something went wrong when solving the system of linear equations. Verify that the chemical equation can be balanced at all, and that it can be balanced uniquely except for multiplication by a constant.")

    normalization_index = get_normalization_index(normalization)
    coeffs /= coeffs[normalization_index]
    coeffs *= normalization_value

    # double-check that the equation is balanced
    if abs(np.sum(mat @ coeffs.reshape(-1,1))) > 1e-10:
        raise RuntimeError('Stoichiometry double-check failed. mat = {}, coeffs = {}'.format(mat, coeffs))

    return list(coeffs[:num_reactants]), list(coeffs[num_reactants:])

def reaction_energy(reactants, products, normalization='r0', unit='hartree'):
    """ 

    Calculates a reaction energy from an unbalanced chemical equation (the equation is first balanced)

    reactants: a list of amsjobs or paths to ams results folders,
    products: a list of amsjobs or paths to ams results folders
    normalization: normalize the chemical equation by setting the corresponding coefficient to 1.
        'r0': first reactant
        'r1': second reactant, ...
        'p0: first product,
        'p1': second product, ...
    unit: Unit of the reaction energy

    Returns: a 3-tuple (coeffs_reactants, coeffs_products, reaction_energy)

    """

    my_reactants = [AMSJob.load_external(x) for x in reactants]
    my_products = [AMSJob.load_external(x) for x in products]
    coeffs_r, coeffs_p = balance_equation(my_reactants, my_products, normalization)
    reaction_energy = None

    energies = [job.results.get_energy(unit=unit) for job in my_reactants+my_products]
    energies = np.array(energies)

    coeffs = np.concatenate(([-x for x in coeffs_r],coeffs_p))
    reaction_energy = np.dot(coeffs, energies)

    return coeffs_r, coeffs_p, reaction_energy

