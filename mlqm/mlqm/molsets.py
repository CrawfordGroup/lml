#!/usr/bin/python3

from . import base
from . import molecule
import json
import numpy as np
import psi4

"""
This module contains some molecule set builders for sets commonly used in this
package.
"""

class PESBuilder(base.MolsetBuilder) :
    """
    Creates a set of geometries based on varying a parameter.
    """

    singleton = None
    
    def __init__(self) :
        pass

    def getsingleton() :
        """
        Implementation of getsingleton.
        """
        if PESBuilder.singleton == None :
            PESBuilder.singleton = PESBuilder()
        return PESBuilder.singleton

    def build(self, inp, **kwargs) :
        """
        Builds the PES set. Can take either a dictionary or a json file name.

        Returns
        -------
        generator
            A generator expression that can be iterated over.
        """
        
        # Read the input file or dictionary.
        if type(inp) is dict :
            dic = inp
        else :
            raise Exception
        
        geom = dic['geom']
        gvars = dic['gvar'] if hasattr(dic['gvar'], "__iter__") else [dic['gvar']]
        dises = dic['dis'] if hasattr(dic['dis'][0], "__iter__") else [dic['dis']]
        pts = dic['pts'] if hasattr(dic['pts'], "__iter__") else [dic['pts']]

        # Generate the ranges.
        peses = [[] for i in range(len(gvars))]
        for gvar, dis, pt, i in zip(gvars, dises, pts, range(len(gvars))) :
            bot = float(dis[0])
            top = float(dis[1])
            if ('pestype' in kwargs and kwargs['pestype'].upper() == 'INCLUDE') or\
               'pestype' not in kwargs :
                peses[i] = np.linspace(bot, top, pt)
            elif 'pestype' in kwargs and kwargs['pestype'].upper() == 'EXCLUDE':
                peses[i] = list(np.linspace(bot, top, pt + 2))
                peses[i].pop(0)
                peses[i].pop(-1)
            else :
                raise Exception

        # Generate the geometries.
        index = [0 for i in range(len(peses))]
        while all(index[i] < len(peses[i]) for i in range(len(peses))) :
            new_geom = geom
            for gvar, i in zip(gvars, range(len(index))) :
                new_geom = new_geom.replace(gvar, str(peses[i][index[i]]))

            mol = psi4.geometry(new_geom)
            mol.move_to_com()
            # Change units to Bohr.
            mol.set_units(psi4.core.GeometryUnits(1))
            geom_array = mol.geometry().to_array()
            out_geom = [tuple([mol.symbol(i), mol.mass(i)] + list(geom_array[i]))
                        for i in range(mol.natom())]

            # Yield the current geometry.
            yield molecule.Molecule(str(hash(tuple(peses[i][index[i]]
                                               for i in range(len(peses))))),
                                         out_geom, mol.molecular_charge(),
                                         mol.multiplicity(), meta =
                                    tuple(peses[i][index[i]] for i in
                                          range(len(peses))))

            # Increment the indices.
            index[0] += 1
            for i in range(len(index) - 1) :
                if index[i] >= pts[i] :
                    index[i] = 0
                    index[i + 1] += 1
        return
            
