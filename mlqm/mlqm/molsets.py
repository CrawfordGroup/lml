#!/usr/bin/python3

from . import base
from . import molecule
import json
import numpy as np
import psi4

"""
molsets

This module contains some molecule set builders for sets commonly used in this
package.

Contributor: Connor Briggs

Classes
-------
PESBuilder
    Builds a PES based on a geometry with variables.
"""

class PESBuilder(base.MolsetBuilder) :
    """
    PESBuilder
    
    Creates a set of geometries based on varying a parameter.

    Contributor: Connor Briggs

    Methods
    -------
    __init__(self)
        Default constructor.

    getsingleton()
        See base.Singleton.getsingleton

    build(self, inp, **kwargs)
        Builds the set of molecules.
    """

    singleton = None
    
    def __init__(self) :
        pass

    def build(self, inp, **kwargs) :
        """
        PESBuilder.build

        Builds the PES set.

        Contributor: Connor Briggs

        Parameters
        ----------
        inp
            The dictionary that specifies the PES. It should contain the fields
            "geom" specifying the geometry in Psi4 notation, "gvar" which
            holds either a single string or a list of strings representing the
            variables in the geometry, "dis" which holds either a list
            containing two numbers, or a list that contains lists of two numbers
            which specifies the ranges to vary the variables over, and "pts"
            which contains either a number or a list of numbers representing
            the number of points to split the ranges into. If only one dis or
            pts is specified for several gvars, then these will be duplicated
            to match the number of gvars.

        pestype
            Either "include" or "exclude", defaults to "include", indicating
            whether the ranges include the bounds or not. Case insensitive.

        Raises
        ------
        TypeError
            If inp is not a dictionary.

        ValueError
            If pestype is passed, but is not "include" or "exclude"

        KeyError
            If one of the fields is missing from inp.

        StopIteration
            When the iteration has been reached.

        Yields
        ------
        Molecule
            A molecule with some geometry in the specified PES.

        Returns
        -------
        generator
            A generator expression that yields the molecules in the PES. Do
            not assume order.
        
        """
        
        # Read the input dictionary.
        if type(inp) is dict :
            dic = inp
        else :
            raise TypeError
        
        geom = dic['geom']
        gvars = dic['gvar'] if hasattr(dic['gvar'], "__iter__") else \
                [dic['gvar']]
        dises = dic['dis'] if hasattr(dic['dis'][0], "__iter__") else \
                [dic['dis'] for i in range(len(gvars))]
        pts = dic['pts'] if hasattr(dic['pts'], "__iter__") else \
              [dic['pts'] for i in range(len(gvars))]

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
                raise ValueError

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
            out_geom = [tuple([mol.symbol(i), int(mol.charge(i)), mol.mass(i)]
                              + list(geom_array[i]))
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

        # Stop the generator.
        raise StopIteration
