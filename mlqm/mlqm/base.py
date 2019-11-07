#!/usr/bin/python3

"""
This will contain class templates and the like. A simple docstring describing
the patterns used, the reasoning behind the use, and other things will be
included in each.

What should be in this file:
    - "Empty" class definitions. Class methods should not have any
        implementation, unless it is simple (e.g. getters/setters). Some
        more complicated methods may be implemented, but the logic behind
        this decision should be stated.

What should not be in this file:
    - Class instantiations. If your class uses one of these patterns, and
        it actually has code, put it in another file. This way, this file
        does not become cluttered with actual code.
    - Functions. Functions should go in another file.

"""

class DatasetBuilder :
    """
    DatasetBuilder

    This class represents a builder, which creates a dataset from some data.
    Implementations of this class should create a Dataset object by implementing
    the 'build' method. This should also be a singleton.
    """

    def __init__(self) :
        pass

    def getsingleton() :
        """
        DatasetBuilder.getsingleton

        This is the heart of a singleton. This returns a reference to a
        single instance of this class. An example implementation is
        provided, with an error at the beginning to make sure this is not
        used. Note that it does not take 'self'. This is because it is a
        method of a class that returns an instance. It assumes that no
        instance has been created yet.

        Returns
        -------
        DatasetBuilder
            Single implementation of the child class.
        """
        raise NotImplemented("Please don't use this template directly")
        if DatasetBuilder.__singleton == None :
            DatasetBuilder.__singleton = DatasetBuilder()
        return DatasetBuilder.__singleton


    def build(self, *args, **kwargs) :
        """
        DatasetBuilder.build

        This is the heart of a builder. It should take some arguments and
        build a Dataset object.

        Returns
        -------
        dict{"setname" : data}
            The dataset representation of the input data.
        """
        raise NotImplemented


    
class InputGenerator :
    """
    InputGenerator

    This template class generates input files. Instances should be specified
    for different programs, since Datasets should have strict implementations.
    Instances should also be singletons, and should override the 'generate'
    method.
    """

    def __init__(self) :
        pass

    def getsingleton() :
        """
        InputGenerator.getsingleton

        See Also
        --------
        DatasetBuilder.getsingleton
        """
        raise NotImplemented

    def generate(self, molset, directory, *args, **kwargs) :
        """
        InputGenerator.generate

        This method should generate the input files for a certain qchem program
        like Psi4 or Gaussian.

        Returns
        -------
        iterable[str]
            A list of the directories created.
        """
        raise NotImplemented

class Runner :
    """
    Runner

    This class template should run a set of files with a specific program.
    Each instance should be a singleton, and should override 'run'.

    """
    def __init__(self) :
        pass

    def getsingleton() :
        """
        Runner.getsingleton

        See Also
        --------
        DatasetBuilder.getsingleton
        """
        raise NotImplemented

    def run(self, *args, **kwargs) :
        """
        Runner.run

        The heart of this class. This should run all the desired files and
        save the results.

        Returns
        -------
        None
        """
        raise NotImplemented

class Model :
    """
    Model

    This class template represents a learned model. It contains a method to
    predict a value and that is it. Teaching should be done in a different
    class.
    """

    def __init__(self) :
        pass

    def predict(self, x, **kwargs) :
        """
        Model.predict

        This method should predict a value based on the represented model.
        The only parameter that is passed for prediction should be the
        input value. Hyperparameters should be passed in to the constructor.

        Returns
        -------
        object
            The predicted value based on inputs.
        """
        raise NotImplemented

class Trainer :
    """
    Trainer
    
    This is a builder, like DatasetBuilder, only the objects it builds are
    models based on a specific type of regression. It should be a singleton,
    and implementations should override the 'train' method.
    """

    def __init__(self) :
        pass

    def getsingleton() :
        """
        Trainer.getsingleton

        See Also
        --------
        DatasetBuilder.getsingleton
        """
        raise NotImplemented

    def train(self, inputs, outputs, **kwargs) :
        """
        Trainer.train

        This should return a trained model based on the regression represented
        by this trainer.

        Returns
        -------
        Model
            The model that represents the data using the method represented by
            this trainer.
        """ 
        raise NotImplemented
         
class MolsetBuilder :
    """
    MolsetBuilder

    This class template represents a builder for a set of molecules. This
    could be, for instance, a builder for a PES based on a certain property,
    or maybe it just reads in geometries. It should be a singleton, and
    instances should override the build method.
    """

    def __init__(self) :
        pass

    def getsingleton() :
        """
        MolsetBuilder.getsingleton

        See Also
        --------
        DatasetBuilder.getsingleton
        """
        raise NotImplemented

    def build(self, *args, **kwargs) :
        """
        MolsetBuilder.build

        This is the method that should be overridden. It should build a set of
        molecules from some input.

        Returns
        -------
        iterable[Molecule]
            A set of geometries.
        """
        raise NotImplemented


