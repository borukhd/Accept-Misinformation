import numpy as np


class OptPars:
    #number of basinhopping iterations per pre_train_person call; note that model evaluation time usually grows exponentially if increased.
    iterations = 200
    parsPerPers = {}
    T = 5
    iterationsFFT = 60 
    Tfft = 0.5


class RandomDisplacementBounds(object):
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds """
        min_step = np.maximum(self.xmin - x, -self.stepsize)
        max_step = np.minimum(self.xmax - x, self.stepsize)

        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step
        return xnew



