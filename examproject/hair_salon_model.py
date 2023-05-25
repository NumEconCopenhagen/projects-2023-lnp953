import numpy as np

class HairSalonModel:
    def __init__(self, eta=0.5, w=1.0):
        """
        Initializing the HairSalonModel class with the given parameters
        
        """
        self.eta = eta
        self.w = w

    def calculate_profit(self, kappa, ell=None):
        """
        Calculating the profit based on the given kappa and ell values.
        
        """
        if ell is None:
            ell = ((1 - self.eta) * kappa / self.w) ** (1 / self.eta)
        
        profit = kappa * ell ** (1 - self.eta) - self.w * ell
        
        return profit


class HairSalonDynamicModel:
    def __init__(self, eta=0.5, w=1.0, rho=0.90, iota=0.01, sigma_epsilon=0.10, R=1.01**(1/12)):
        """
        Initializing the HairSalonDynamicModel class with the given parameters
        
        """
        self.eta = eta
        self.w = w
        self.rho = rho
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R

    def calculate_profit(self, kappa, ell=None):
        if ell is None:
            ell = ((1 - self.eta) * kappa / self.w) ** (1 / self.eta)
        profit = kappa * ell ** (1 - self.eta) - self.w * ell
        return profit

    def calculate_h(self, epsilon_series):
        """
        Calculating the ex post value of the salon based on the epsilon series
        
        """
        #Calculating kappa_t using AR(1)
        kappa_series = np.exp(self.rho * np.concatenate(([0], epsilon_series[:-1]))) 

        #Calculating ell_t using the policy 
        ell_series = ((1 - self.eta) * kappa_series / self.w) ** (1 / self.eta)  

        #Initializing an array to store profits for each time step
        profit_series = np.zeros_like(kappa_series) 

        #Initializing an array to store adjustment costs for each time step 
        adjustment_cost_series = np.zeros_like(kappa_series) 

        for t in range(len(kappa_series)):
            #Calculating profit for each time step
            profit_series[t] = self.calculate_profit(kappa_series[t], ell_series[t])
            #Checing if ell changes from the time step before 
            if t > 0 and ell_series[t] != ell_series[t-1]:
                #Assigning adjustment cost if there is a change in ell
                adjustment_cost_series[t] = self.iota 

        #Calculating discounted profits using the discount factor R
        discounted_profits = self.R**(-np.arange(len(kappa_series))) * (profit_series - adjustment_cost_series)

        #Summing the discounted profits to get the ex post value of the salon
        h = np.sum(discounted_profits)  
        return h

