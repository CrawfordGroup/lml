#!/usr/bin/python3

class Molecule :
    """
    Molecule
    
    Represents a molecule with the goal of separating the interface from the
    program used to parse it. It will have a name which will be used for things
    like naming directories for inputs.
    """

    def __init__(self, name, geometry, charge, spin, meta = None) :
        """
        Initialize a molecule. Geometry should be a list where each element is
        of the form
        (atomic_symb, mass, x, y, z)
        The 'meta' parameter passes other data to store with this molecule.
        This allows for things like storing snapshot numbers or PES coordinates.
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
        Return the coordinate geometry of the molecule. Each entry should be
        of the form
        (atom_symb, mass, x, y, z)

        Returns
        -------
        list
            List of the cordinates and other things.
        """
        return self.__geometry

    def charge(self) :
        """
        Return the total formal charge on the molecule.

        Returns
        -------
        int
            The total charge on the molecule.
        """
        return self.__charge

    def spinstate(self) :
        """
        Returns the spin state of the molecule.

        Returns
        -------
        int
            Spin state of the molecule.
        """
        return self.__spin

    def metadata(self) :
        """
        Returns any extra data associated with this molecule.

        Returns
        -------
        ?
            Any extra data.
        """
        return self.__meta
        
    def __str__(self) :
        """
        Molecule.getname

        This is a name unique to the molecule. An example could be the value of
        the variable parameter for a PES.

        Returns
        -------
        str
            The name of the molecule.
        """
        return self.__name
