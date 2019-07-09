'''
functions to facilitate theta estimation
'''

import numpy as np 
from scipy.optimize import minimize 
from scipy.stats import lognorm, norm 
from scipy.special import expit 


def theta_fn(difficulties, student_prior, response_pattern):
    def fn(theta):
        theta = theta[0] 
        probabilities = expit(theta - difficulties)
        #print(probabilities) 
        log_likelihood = student_prior.logpdf(theta) 
        for i, rp in enumerate(response_pattern):
            log_likelihood += np.log1p((2 * probabilities[i] - 1) * rp) 
        #print(log_likelihood)
        return  -log_likelihood 
    return fn 


def calculate_theta(difficulties, response_pattern):
    """
    given learned item difficulties and a model response pattern, estimate theta
    """ 

    student_prior = norm(loc=0., scale=1.) 
    fn = theta_fn(difficulties, student_prior, response_pattern) 
    result = minimize(fn, [-2], method='Nelder-Mead') 
    return result['x']


def test():
    diffs = [-3., -2., 2., 3.]
    rp = [1,1,1,1]

    print(calculate_theta(diffs, rp)) 
