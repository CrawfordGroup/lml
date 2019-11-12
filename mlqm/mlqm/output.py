#!/usr/bin/python3

"""
output

Holds classes that make the program output a bit more aesthetic.

Contributor: Connor Briggs

Classes
-------
ProgressBar
    Represents a progress bar.

OutputMediator
    Holds output things, like progress bars, and allows for interactions
    between them

ProgressBarCommand
    Represents a command to a progress bar

EndMediatorCommand
    A command to end the OutputMediator's execution.

Observer
    An abstract observer class.

Outputer
    Outputs a line by command from the mediator.

OutputCommand
    Command to output a line.
"""

import threading
import time
from . import base
import atexit

class OutputMediator(base.Singleton) :
    """
    OutputMediator

    Mediates the interactions between objects to be output. It's a singleton,
    since there is really only one output stream in the cases where it's needed.
    This runs on its own thread.

    Contributor: Connor Briggs

    Fields
    ------
    __queue
        Commands that have been queued.

    __thread
        The thread that this runs on.

    __queue_lock
        A lock for the queue so that interactions are more predictable.

    __listeners
        A dictionary of listeners and what kind of commands these are seeking.

    Methods
    -------
    __init__(self)
        Initialize all the things needed for this to run and then start it.

    submit(self, command)
        Submit a command to run. May block if the queue is being used.

    __run(self)
        Runs the loop to process commands.

    register(self, command_type, listener)
        Register a listener for a type of command.

    deregister(self, listener)
        Remove a listener from the list.

    getsingleton()
        See base.Singleton.getsingleton
    """
    def __init__(self) :
        self.__queue = []
        # Daemon so that the thread ends when the program does.
        self.__thread = threading.Thread(target = self.__run, daemon = True)
        self.__listeners = {}
        self.__queue_lock = threading.Lock()
        self.__thread.start()

    def submit(self, command) :
        """
        submit

        Submits a command to the command queue to run.

        Contributor: Connor Briggs
        
        Parameters
        ----------
        command
            The command to submit.
        """
        self.__queue_lock.acquire()
        self.__queue.append(command)
        self.__queue_lock.release()

    def __run(self) :
        """
        __run

        The main loop of the command interpreter. Takes commands off the queue
        and then sends them to the right places, and then sleeps.

        Contributor: Connor Briggs
        """

        while True :
            # Obtain a lock on the queue.
            with self.__queue_lock :
                while len(self.__queue) > 0 :
                    # Pop the first item added.
                    command = self.__queue.pop(0)
                    # If we are telling this to stop, then stop.
                    if type(command) == EndMediatorCommand :
                        return
                    # If we are being told to now register or deregister, do so.
                    if type(command) == RegDeregCommand :
                        self.__queue_lock.release()
                        if command.state :
                            self.register(command.commands(),
                                          command.listener())
                        else :
                            self.deregister(command.listener())
                        self.__queue_lock.acquire()
                        continue
                    # Get the listeners for this kind of command.
                    try :
                        listeners = self.__listeners[type(command)]
                    except :
                        continue
                    for l in listeners :
                        # Process the command.
                        l.process(command)
            # Release this thread to the system for 0.1 seconds.
            time.sleep(0.1)
        return

    def register(self, command_type, listener) :
        """
        register

        Registers a listener for a specific command type.

        Contributor: Connor Briggs

        Parameters
        ----------
        command_type
            Either a command class, an instance of a command class, or a list
            of these to which to assign this listener.

        listener
            The listener to add.

        Returns
        -------
        True if the addition was successful, False otherwise.
        """
        # We want to get a lock on this, since adding a listener in the middle
        # of processing could be disasterous.
        with self.__queue_lock :
            if type(command_type) is type :
                # Make sure there is a spot for the listener.
                if command_type not in self.__listeners :
                    self.__listeners[command_type] = []
                # Don't add duplicates.
                if listener in self.__listeners[command_type] :
                    return False
                self.__listeners[command_type].append(listener)
            elif hasattr(command_type, "__iter__") :
                for c in command_type :
                    if type(command_type) is type :
                        if command_type not in self.__listeners :
                            self.__listeners[command_type] = []
                        if listener in self.__listeners[command_type] :
                            return False
                        self.__listeners[command_type].append(listener)
                    else :
                        if type(command_type) not in self.__listeners :
                            self.__listeners[type(command_type)] = []
                        if listener in self.__listeners[type(command_type)] :
                            return False
                        self.__listeners[type(command_type)].append(listener)
            else :
                if type(command_type) not in self.__listeners :
                    self.__listeners[type(command_type)] = []
                if listener in self.__listeners[type(command_type)] :
                    return False
                self.__listeners[type(command_type)].append(listener)
            return True

        
    def deregister(self, listener) :
        """
        OutputMediator.deregister

        Removes a listener from the listener pool.

        Contributor: Connor Briggs

        Parameters
        ----------
        listener
            The listener to remove.

        """
        with self.__queue_lock :
            for k in self.__listeners :
                self.__listeners[k].remove(listener)
        

class Observer :
    """
    Observer
    
    Represents an observer object. These change when their subject changes.

    Contributor: Connor Briggs

    Methods
    -------
    __init__(self)
        Default constructor

    process(self, command) @override
        Process a command.
    """
    def __init__(self) :
        pass

    def process(self, command) :
        """
        Observer.process

        Process a command.
        """
        raise NotImplemented

class ProgressBar(Observer) :
    """
    ProgressBar

    Represents a progress bar. This takes commands such as ProgressBarCommands
    and OutputCommands.

    Contributor: Connor Briggs

    Attributes
    ----------
    __size
        The size of the progress bar.

    __max_items
        The maximum number of items represented.

    __curr_items
        The current number of items. With __max_items, this is how the
        completeness of the progress bar is determined.
        
    __length
        The length of the last line.

    Methods
    -------
    __init__(self, max_items, size = 50)
        Constructor.
    process(self, command)
        Process a command.

    add_item(self)
        Add one to the item counter.

    write_bar(self)
        Write the progress bar.

    clear_bar(self)
        Overwrite the progress bar with spaces so that it doesn't stick out
        at the end of a line.

    skip(self)
        Go to the next line and keep the current state of the line.
    """
    def __init__(self, max_items, size = 50) :
        self.__size = size
        self.__max_items = max_items
        self.__curr_items = 0
        self.__length = 0

    def process(self, command) :
        """
        ProgressBar.process

        Processes a command.

        Contributor: Connor Briggs

        Parameters
        ----------
        command
            The command to process.
        """
        # If we get a ProgressBarCommand, do what is in the table.
        if type(command) == ProgressBarCommand :
            {"inc": lambda : self.add_item(),
             "write": lambda : self.write_bar(),
             "clear": lambda : self.clear_bar(),
             "skip": lambda : self.skip()}[command.action()]()
        if type(command) == OutputCommand :
            self.skip()

    def add_item(self) :
        self.__curr_items += 1

    def write_bar(self) :
        """
        ProgressBar.write_bar

        Write the progress bar to stdout.

        Contributor: Connor Briggs
        """
        # Clear the bar first if needed.
        if self.__length != 0 :
            self.clear_bar()
        # Draw the bar.
        hashes = ''.join('#' for i in range(int(
            self.__curr_items / self.__max_items * self.__size)))
        
        outs = (f"[%{-self.__size}s] %d%%")%(hashes, int(self.__curr_items /
                                                          self.__max_items *
                                                          100))
        self.__length = len(outs)
        print(outs, end = '', flush = True)

    def clear_bar(self) :
        """
        ProgressBar.clear_bar

        Clears the current progress bar.

        Contributor: Connor Briggs
        """
        print(f"\r{''.join(' ' for i in range(self.__length + 1))}\r",
              end = "", flush = True)
        self.__length = 0

    def skip(self) :
        """
        ProgressBar.skip

        Goes to the next line without clearing the current progress bar.

        Contributor: Connor Briggs
        """
        if self.__length != 0 :
            self.__length = 0
            print("", flush = True)
        
class ProgressBarCommand :
    """
    ProgressBarCommand
    
    Represents a command for a progress bar.

    Contributor: Connor Briggs

    Attributes
    ----------
    __act
        The name of the action to take.

    Methods
    -------
    __init__(self, act)
        Constructor

    action(self)
        Returns the action that should be taken.

    """
    def __init__(self, act) :
        self.__act = act.lower()

    def action(self) :
        return self.__act

class EndMediatorCommand :
    """
    EndMediatorCommand

    Empty class to tell the mediator to stop.
    """
    pass

class Outputter(Observer, base.Singleton) :
    """
    Outputter

    Outputs a line by a command. This is made so that it works with the
    ProgressBar class so that race conditions are not a thing.

    Contributor: Connor Briggs

    Attributes
    ----------
    __singleton
        Singleton reference.

    Methods
    -------
    __init__(self)
        Constructor

    process(self, command)
        Process a command.

    getsingleton()
        See base.DatasetBuilder.getsingleton
    """
    
    def __init__(self) :
        pass

    def process(self, command) :
        print(str(command), flush = True)


class OutputCommand :
    """
    OutputCommand

    Represents a command to the line outputter.

    Contributor: Connor Briggs

    Attributes
    ----------
    __line
        The line to write.

    Methods
    -------
    __init__(self, line)
        Constructor.

    __str__(self)
        Get the string from this.
    """
    def __init__(self, line) :
        self.__line = line

    def __str__(self) :
        return self.__line

class RegDeregCommand :
    """
    RegDeregCommand
    
    Represents a registration or deregistration command for the mediator. This
    is to make sure that all the actions that you want to have act on a listener
    happen before it is removed, or when you want a set of commands in the queue
    to not act on a listener from before it was added.

    Contributor: Connor Briggs

    Methods
    -------
    __init__(self, state, comms, listen)
        Constructor. If state is true, register. If it is false, deregister.
    state(self)
        Whether to register or deregister
    listener(self)
        What is the listener.
    commands(self)
        Which commands to register to. Not used for deregister.
    """
    def __init__(self, state, comms, listen) :
        self.__state = state
        self.__listen = listen
        self.__comms = comms

    def state(self) :
        return self.__state

    def listener(self) :
        return self.__listen

    def commands(self) :
        return self.__comms
