#!/usr/bin/python3

"""
base

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

Contributor: Connor Briggs

Classes
-------
Singleton
    Template for classes which should have only one instance at any one time.
    
DatasetBuilder
    Represents an abstract builder for datasets.

InputGenerator
    Represents an abstract generator that generates input files for
    a specific QC program.

Runner
    Represents a strategy for running input files throug a QC program.

Model
    Represents an abstract regression model.

Trainer
    Represents an abstract training method for a regression.

MolsetBuilder
    Represents a way to make a set of molecules.
"""

import concurrent.futures
import os

class Singleton :
    """
    Singleton

    The singleton pattern is a very useful pattern, and can be applied in many
    places. It's purpose is to provide exactly one instance of its class. In
    this way, things like factories and mediators don't need to be instanced
    and have the one instance passed around. Instead, an instance can be
    retrieved from this class, and the class and instance are more or less
    bound. If a piece of code asks for an instance of a singleton class, it
    will return the same reference as it would to any other piece of code in
    the program.

    Contributor: Connor Briggs

    Attributes
    ----------
    __singleton
        The single unique instance of the class.

    Methods
    -------
    __init__(self)
        Default constructor.

    getsingleton
        Returns the single instance of the class. If it is not initialized, it
        will be initialized here.
    """
    __singleton = None

    def __init__(self) :
        pass

    @classmethod
    def getsingleton(cls) :
        """
        Singleton.getsingleton

        This is the heart of a singleton. This returns a reference to a
        single instance of this class. 

        Contributor: Connor Briggs

        Returns
        -------
        Single implementation of the child class.
        """
        if cls.__singleton == None :
            cls.__singleton = cls()
        return cls.__singleton

class DatasetBuilder(Singleton) :
    """
    DatasetBuilder

    This class represents a builder, which creates a dataset from some data.
    Implementations of this class should create a Dataset object by implementing
    the 'build' method. This should also be a singleton.

    Contributor: Connor Briggs

    Methods
    -------
    __init__(self)
        A do-nothing default constructor.
        
    getsingleton()
        See Singleton.getsingleton()
        
    build(self, *args, **kwargs) @override
        Returns a dataset built by this object using the given arguments.
    """

    def __init__(self) :
        pass

    def build(self, *args, **kwargs) :
        """
        DatasetBuilder.build

        This is the heart of a builder. It should take some arguments and
        build a Dataset object.

        Contributor: Connor Briggs

        Parameters
        ----------
        self
            A reference to the single unique builder.
            
        args, kwargs
            These will depend on implementation.

        Raises
        ------
        NotImplemented
            Subclasses of this should be used. This has no code.

        Returns
        -------
        dict{"setname" : data}
            The dataset representation of the input data.
        """
        raise NotImplemented


    
class InputGenerator(Singleton) :
    """
    InputGenerator

    This template class generates input files. Instances should be specified
    for different programs, since Datasets should have strict implementations.
    Instances should also be singletons, and should override the 'generate'
    method.

    Contributor: Connor Briggs

    Methods
    -------
    __init__(self)
        Do-nothing default constructor.

    getsingleton()
        See Singleton.getsingleton

    generate(self, molset, directory, *args, **kwargs) @override
        Generate the input files in the specified directory for the given
        set of molecules.
    """

    def __init__(self) :
        pass

    def generate(self, molset, directory, *args, **kwargs) :
        """
        InputGenerator.generate

        This method should generate the input files for a certain qchem program
        like Psi4 or Gaussian.

        Contributor: Connor Briggs

        Paramters
        ---------
        self
            A reference to the single instance of this class.
            
        molset
            An iterable set of molecules. The Molecule.__str__ method should
            be used as names for data directories.

        directory
            The directory to place all the generated input files.

        args, kwargs
            Depends on implementation.

        Raises
        ------
        NotImplemented
            Children should implement this method.

        Returns
        -------
        iterable[str]
            A list of the directories created.
        """
        raise NotImplemented

# I believe that this needs to come here, or at least after the definition of
# base.Singleton, since output.OutputMediator is a Singleton. Python doesn't
# like it if this is included before everything it needs is defined.
from . import output

class Runner(Singleton) :
    """
    Runner

    This class template should run a set of files with a specific program.
    Each instance should be a singleton, and should override 'run'.

    Contributor: Connor Briggs

    Methods
    -------
    __init__(self)
        Do-nothing default constructor.
        
    getsingleton()
        See Singleton.getsingleton()

    run_item(self, *args, **kwargs) @override
        Runs an input file through the program.

    run(self, directories, inpf, outpf, *args, **kwargs)
        Goes through a directory and runs each input file.

    """
    def __init__(self) :
        pass

    def run_item(self, inpf, outpf, *args, **kwargs) :
        """
        Runner.run_item

        Run the single item throug a given program. This should be overriden
        by the child.

        Contributor: Connor Briggs

        Paramters
        ---------
        inpf
            The name of the input file, including the path.

        outpf
            The name of the output file, including the path.

        Raises
        ------
        NotImplemented
            Children should implement this, but it's not implemented here.
        """
        raise NotImplemented
        
    def run(self, directories, inpf, outpf, executor = None, *args, **kwargs) :
        """
        Runner.run

        Run the inputs in the directories using a given program.
        This method is implemented here since it is general to this class. It
        can still be overridden if needed, but the default implementation is
        provided so that children do not need to have their own implementations.
        Does not return.

        Contributor: Connor Briggs

        Paramters
        ---------
        directories
            An iterable set of directories to look for inputs.

        inpf
            The name of the input files.

        outpf
            The name of the output files.
            
        executor = None
            A concurrent.futures.Executor instance to use to execute these
            jobs in parallel, or None if the jobs should be executed
            sequentially.

        regen = False
            Whether to re-run the inputs if there is already an output file.

        mediator = None
            The output mediator, if desired.

        bar_len
            The length of the progress bar, if desired.
            
        args, kwargs
            Passed to run_item.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.

        TypeError
            If executor is not an Executor or None.
        """
        # Progress bar stuff.
        bar = None
        if 'mediator' in kwargs :
            if 'bar_len' in kwargs :
                bar = output.ProgressBar(len(directories), size =
                                              kwargs['bar_len'])
            else :
                bar = output.ProgressBar(len(directories))
            kwargs['mediator'].register(output.ProgressBarCommand, bar)
            kwargs['mediator'].register(output.OutputCommand, bar)
            kwargs['mediator'].submit(output.ProgressBarCommand("write"))
        
        # First case: If there is no executor, then execute sequentially.
        if executor == None :
            for d in directories :
                # Output already exists. Skip it if requested.
                if os.path.isfile(d + "/" + outpf) and ("regen" not in kwargs
                                                        or not kwargs['regen']):
                    continue
                # Input does not exist!
                if not os.path.isfile(d + "/" + inpf) :
                    raise FileNotFoundError

                #Run the input file.
                self.run_item(d + "/" + inpf, d + "/" + outpf, *args, **kwargs)
        # Second case: If the program should be run in parallel.
        elif isinstance(executor, concurrent.futures.Executor) :
            futures = []
            for d in directories :
                # Output already exists. Skip it if requested.
                if os.path.isfile(d + "/" + outpf) and ("regen" not in kwargs
                                                        or not kwargs['regen']):
                    continue
                # Input does not exist!
                if not os.path.isfile(d + "/" + inpf) :
                    raise FileNotFoundError
                # Add the file to the queue.
                futures.append(executor.submit(self.run_item, d + "/" + inpf,
                                               d + "/" + outpf, *args, **kwargs))
            # Run the queue.
            concurrent.futures.wait(futures)
        else :
            # executor is not of an expected type.
            raise TypeError
        
        # More progress bar stuff.
        if 'mediator' in kwargs :
            kwargs['mediator'].submit(output.ProgressBarCommand("skip"))
            kwargs['mediator'].submit(output.RegDeregCommand(False, None,
                                                                  bar))
            
class Model :
    """
    Model

    This class template represents a learned model. It contains a method to
    predict a value and that is it. Teaching should be done in a different
    class.

    Contributor: Connor Briggs

    Methods
    -------
    __init__(self)
        Default constructor.

    predict(self, x, **kwargs) @override
        Find the predicted value from the given input.
    """

    def __init__(self) :
        pass

    def predict(self, x, **kwargs) :
        """
        Model.predict

        This method should predict a value based on the represented model.
        The only parameter that is passed for prediction should be the
        input value. Hyperparameters should be passed in to the constructor.

        Contributor: Connor Briggs

        Paramters
        ---------
        x
            The input value to predict from.

        kwargs
            Specified by children.

        Raises
        ------
        NotImplemented
            Should be implemented by children, but not here.
            
        Returns
        -------
        object
            The predicted value based on inputs. The type should be defined
            by children.
        """
        raise NotImplemented

class Trainer(Singleton) :
    """
    Trainer
    
    This is a builder, like DatasetBuilder, only the objects it builds are
    models based on a specific type of regression. It should be a singleton.

    Contributor: Connor Briggs

    Methods
    -------
    __init__(self)
        Default constructor

    getsingleton()
        See Singleton.getsingleton()

    train(self, inputs, outputs, **kwargs)
        Train a model using the given inputs and outputs.
    """

    def __init__(self) :
        pass

    def train(self, inputs, outputs, **kwargs) :
        """
        Trainer.train

        This should return a trained model based on the regression represented
        by this trainer.

        Contributor: Connor Briggs

        Parameters
        ----------
        inputs
            The set of inputs to use to train.

        outputs
            The set of corresponding outputs to the inputs to use to train.

        Raises
        ------
        NotImplemented
            Should be implemented by children.

        Returns
        -------
        Model
            The model that represents the data using the method represented by
            this trainer.
        """ 
        raise NotImplemented
         
class MolsetBuilder(Singleton) :
    """
    MolsetBuilder

    This class template represents a builder for a set of molecules. This
    could be, for instance, a builder for a PES based on a certain property,
    or maybe it just reads in geometries. It should be a singleton, and
    instances should override the build method.

    Contributor: Connor Briggs

    Methods
    -------
    __init__(self)
        Default constructor.

    getsingleton()
        See Singleton.getsingleton()

    build(self, *args, **kwargs) @override
        Build the set of molecules.
    """

    def __init__(self) :
        pass

    def build(self, *args, **kwargs) :
        """
        MolsetBuilder.build

        This is the method that should be overridden. It should build a set of
        molecules from some input.

        Contributor: Connor Briggs

        Raises
        ------
        NotImplemented
            Should be implemented by children.

        Returns
        -------
        iterable[Molecule]
            A set of geometries.
        """
        raise NotImplemented
