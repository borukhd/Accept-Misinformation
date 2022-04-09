#adjust import structure if started as script
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


""" 
News Item Processing model implementation.
"""
import ccobra
import random
import math

class BaselineRandom(ccobra.CCobraModel):
    """ News reasoning CCOBRA implementation.
    """
    def __init__(self, name='BaselineRandom', commands = []):
        """ Initializes the news reasoning model.
        Parameters
        ----------
        name : str
            Unique name of the model. Will be used throughout the ORCA
            framework as a means for identifying the model.
        """
        self.parameter = {}
        self.sorted_par_keys = sorted(self.parameter.keys())
        super().__init__(name, ['misinformation'], ['single-choice'])

    def predict(self, item, **kwargs):
        return random.choice(item.choices)

    def predictS(self, item, **kwargs):
        if len(kwargs.keys()) == 1:
            kwargs = kwargs['kwargs']
        return 0.5

    def adapt(self, item, target, **kwargs):
        pass

    def adaptS(self, itemPair):
        pass
        
