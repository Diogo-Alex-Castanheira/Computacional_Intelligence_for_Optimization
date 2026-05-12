from abc import ABC, abstractmethod


class Solution(ABC):
    
    def __init__(self, repr = None):
        # To initialize a solution we need to know its representation.
        # If no representation is given, a solution is randomly initialized.
        if repr is None:
            repr = self.random_initial_representation()
        # Attributes
        
        # super().__init__(repr=repr)
        self._fitness = None  # Cache fitness value, to avoid recomputation
        self.repr = repr
        
    # Method that is called when we run: print(object_of_the_class)
    def __repr__(self):
        return str( self.repr)

    # abstract methods will need to be implemented in the problem-specific (sub)classes,
    # so that objects can be instantiated from those classes.
    @abstractmethod
    def fitness(self):
        pass
    
    @abstractmethod
    def random_initial_representation(self):
        pass
    
    @abstractmethod
    def with_repr(self, new_repr):
        """Return a new instance with the given representation, preserving problem-specific parameters."""
        pass