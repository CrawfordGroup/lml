#!/usr/bin/python3

"""
molecule

Contains a definition of Molecule in order to separate the internal
representation of the molecule from the QC program.

Contributor: Connor Briggs

Classes
-------
Molecule
    Represents a molecule.
"""

class Molecule :
    """
    Molecule
    
    Represents a molecule with the goal of separating the interface from the
    program used to parse it. It will have a name which will be used for things
    like naming directories for inputs.

    Contributor: Connor Briggs

    Attributes
    ----------
    __name: str
        The name of this molecule to be used to name files and the like.

    __geometry: list
        Specification of the geometry of the molecule. Each entry in the list
        will look like (atomic_symbol, atomic_num, mass, x, y, z).

    __charge: int
        The total charge on the molecule.

    __spin: int
        The spin multiplicity of the molecule.

    __meta
        Any metadata that you want to associate to this molecule.

    Methods
    -------
    __init__(self, name, geometry, charge, spin, meta = None)
        Constructor

    geometry
    spinstate
    charge
    metadata
        Getters for these values.

    __str__
        Returns the name.
    """

    def __init__(self, name, geometry, charge = 0, spin = 1, meta = None) :
        """
        Molecule.__init__
        
        Initialize a molecule. Geometry should be a list where each element is
        of the form
        (atomic_symb, atomic_num, mass, x, y, z)
        The 'meta' parameter passes other data to store with this molecule.
        This allows for things like storing snapshot numbers or PES coordinates.

        Contributor: Connor Briggs

        Parameters
        ---------
        name
            The name of the molecule.

        geometry
            The geometry of the molecule. It should be in the form of
            (atomic_symb, atomic_number, mass, x, y, z), where atomic_symb is
            the atomic symbol of the atom at this point, atomic_number is the
            atomic number of the atom at this point, mass is the mass in
            atomic units at the point, and x, y, and z are the coordinates of
            the atom in Ångstrom.

        charge
            The total charge on the molecule in electron charges.

        spin
            The spin multiplicity of the molecule.

        meta = None
            Any metadata that you want to associate with this molecule.
        """
        self.__name = name
        self.__geometry = []
        self.__charge = charge
        self.__spin = spin
        self.__meta = meta

        for r in geometry :
            self.__geometry.append(r)

    def geometry(self) :
        """
        Molecule.geometry
        
        Return the coordinate geometry of the molecule. Each entry should be
        of the form
        (atom_symb, mass, x, y, z), where atom_symb is the atomic symbol of the
        atom at this position, mass is the mass of the atom in amu, and x, y,
        and z are the coordinates in Bohr.

        Contributor: Connor Briggs

        Returns
        -------
        list
            List of the cordinates and other things.
        """
        return self.__geometry

    def charge(self) :
        """
        Molecule.charge

        Return the total formal charge on the molecule.

        Contributor: Connor Briggs

        Returns
        -------
        int
            The total charge on the molecule.
        """
        return self.__charge

    def spinstate(self) :
        """
        Molecule.spinstate

        Returns the spin state of the molecule.

        Contributor: Connor Briggs

        Returns
        -------
        int
            Spin state of the molecule.
        """
        return self.__spin

    def metadata(self) :
        """
        Molecule.metadata
        
        Returns any extra data associated with this molecule.

        Contributor: Connor Briggs

        Returns
        -------
        object
            Any extra data.
        """
        return self.__meta
        
    def __str__(self) :
        """
        Molecule.getname

        This is a name unique to the molecule. An example could be the value of
        the variable parameter for a PES.

        Contributor: Connor Briggs

        Returns
        -------
        str
            The name of the molecule.
        """
        return self.__name
    
