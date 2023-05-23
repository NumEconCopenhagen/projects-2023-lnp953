
from types import SimpleNamespace

import numpy as np
from scipy import optimize


import pandas as pd 
import matplotlib.pyplot as plt

class HouseholdSpecializationModelClass:

    def __init__(self):
        """Initialize the model and set up parameters"""

        # a. create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # b. preferences baseline parameters
        par.rho = 2.0 
        par.nu = 0.001
        par.epsilon_m = 1.0
        par.epsilon_f = 1.0
        par.epsilon = 1.0
        par.omega = 0.5 

        #Emasculating preference parameter
        par.delta = 1

        # c. household production baseline parameters
        par.alpha = 0.5
        par.sigma = 1.0

        # d. wages baseline parameters
        par.wM = 1.0
        par.wF = 1.0

        #Creating a vector for the female's wage. 
        #The start of the interval is 0.8, the end is 1.2 and the number of items to generate within the range is 5
        par.wF_vec = np.linspace(0.8,1.2,5)

    
        # e. targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # f. solution
        #Vectors that stores optimal male/female low/high-skilled hours
        sol.LM_vec = np.zeros(par.wF_vec.size) 
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan #Estimated beta0 in regression
        sol.beta1 = np.nan #Estimated beta1 in regression

    def calc_utility(self,LM,HM,LF,HF):
        """Calculate utility """

        par = self.par
        sol = self.sol

        # a. consumption of market goods
        C = par.wM*LM + par.wF*LF

        # b. home production
        if par.sigma == 1:
            H = HM**(1-par.alpha)*HF**par.alpha
        elif par.sigma == 0:
            H = min(HM,HF)
        else:
            H = ((1-par.alpha)*HM**((par.sigma-1)/par.sigma)+par.alpha*HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # c. total consumption utility
        Q = C**par.omega*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # d. disutlity of work
        epsilon_m = 1+1/par.epsilon_m
        epsilon_f = 1+1/par.epsilon_f
        TM = LM+HM
        TF = LF+HF
        if par.delta ==1:
            disutility = par.nu*(TM**epsilon_m/epsilon_m+TF**epsilon_f/epsilon_f)
        else:
            disutility = par.nu*((LM+par.delta*HM)**epsilon_m/epsilon_m+TF**epsilon_f/epsilon_f)
        
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """Solve model discretely.

        Args:
            do_print (bool): Whether to print the results.

        Returns:
            opt (SimpleNamespace): Optimal solution.

        """
        
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # a. all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) #All combinations
    
        LM = LM.ravel() #Vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # b. calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # c. set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # d. find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

    

        # e. print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt

    def solve(self,do_print=False):
        """Solving the model continuously.

        Args:
            do_print (bool): Whether to print the results.

        Returns:
            opt (SimpleNamespace): Optimal solution.

        """
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()

        #Mimimizing:
        def obj(x):
            LM, HM, LF, HF = x
            return -self.calc_utility(LM, HM, LF, HF)

        #Constraints:
        def constraint1(x):
            LM, HM, LF, HF = x
            return 24-(LM+HM)

        def constraint2(x):
            LM, HM, LF, HF = x
            return 24-(LF+HF)

        #Setting an array for constraints:
        constraints = [{"type":"ineq", "fun":constraint1}, {"type":"ineq", "fun":constraint2}]   


        #Initial guess:
        x0 = [12, 12, 12, 12,]

        #Solving
        result = optimize.minimize(obj, x0, constraints=constraints, method = "SLSQP", tol= 1e-08)

        #Saving results:
        opt.LM = sol.LM = result.x[0]
        opt.HM = sol.HM = result.x[1]
        opt.LF = sol.LF = result.x[2]
        opt.HF = sol.HF = result.x[3]

        #printing:
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')
        return opt


    def solve_wF_vec(self,discrete=False):
        """Solve model for a vector of female wages.

        Args:
            discrete (bool): Whether to solve discretely.

        """
        par = self.par
        sol = self.sol

        #Looping over female wages
        for i, wF in enumerate(par.wF_vec):

            par.wF = wF

            #solving:
            if discrete:
                opt = self.solve_discrete()
            else:
                opt= self.solve()

            #store results
            sol.LF_vec[i] = opt.LF
            sol.HF_vec[i] = opt.HF
            sol.LM_vec[i] = opt.LM
            sol.HM_vec[i] = opt.HM
    
    #Running regression:
        self.run_regression()


    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    

    def estimate(self, params=None):
        """Estimating sigma and delta values based on the model and return the error value."""
        par = self.par
        
        #Setting the parameter values if provided
        if params is not None:
            if 'sigma' in params:
                par.sigma = params['sigma']
            if 'delta' in params:
                par.delta = params['delta']
        
        #Defining the objective function for estimation
        def objective(params):
            par.sigma, par.delta = params
            par.alpha = 0.5  #Ensuring alpha is 0.5
            opt = self.solve()
            return np.abs(opt.HF - self.sol.HF_vec).sum()  #The error value
        
        #Initial guess for sigma and delta
        initial_params = [par.sigma, par.delta]
        
        #Optimizing using the Nelder-Mead method to estimate sigma and delta
        result = optimize.minimize(objective, initial_params, method='Nelder-Mead')
        
        #Updating the sigma and delta values with the estimated values
        par.sigma, par.delta = result.x
        
        #Calling the estimated values of sigma and delta
        estimated_sigma = par.sigma
        estimated_delta = par.delta
        
        #Returning the estimated values of sigma and delta as well as the error value
        return estimated_sigma, estimated_delta, result.fun







