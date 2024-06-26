import numpy as np
import pandas as pd

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

            #Checking if ell changes from the time step before 
            if t > 0 and ell_series[t] != ell_series[t-1]:
                
                #Assigning adjustment cost if there is a change in ell
                adjustment_cost_series[t] = self.iota 

        #Calculating discounted profits using the discount factor R
        discounted_profits = self.R**(-np.arange(len(kappa_series))) * (profit_series - adjustment_cost_series)

        #Summing the discounted profits to get the ex post value of the salon
        h = np.sum(discounted_profits)  
        return h

class HairSalonPolicyModel(HairSalonDynamicModel):
    def __init__(self, delta=0.05, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def calculate_ell_ast(self, kappa):

        #Calculating the optimal ell value based the kappa value
        return ((1 - self.eta) * kappa / self.w) ** (1 / self.eta)

    def calculate_ell_t(self, ell_prev, kappa):

        # Calculating the ell value at time t based on the previous ell value and kappa value
        ell_ast = self.calculate_ell_ast(kappa)
        if abs(ell_prev - ell_ast) > self.delta:
            return ell_ast
        else:
            return ell_prev

    def calculate_h_with_policy(self, epsilon_series):
        #Calculating the ex post value of the salon based on the epsilon series
        
        kappa_series = np.exp(self.rho * np.concatenate(([0], epsilon_series[:-1])))

        #Initializing ell series
        ell_series = np.zeros_like(kappa_series) 

        #Initializing profit series 
        profit_series = np.zeros_like(kappa_series)

        #Initializing adjustment cost series
        adjustment_cost_series = np.zeros_like(kappa_series)  

        for t in range(len(kappa_series)):

            #Calculating ell value for each time step
            ell_series[t] = self.calculate_ell_t(ell_series[t-1], kappa_series[t]) 

            #Calculating profit for each time step
            profit_series[t] = self.calculate_profit(kappa_series[t], ell_series[t])  
            if t > 0 and ell_series[t] != ell_series[t-1]:

                #Changing adjustment cost if ell changes
                adjustment_cost_series[t] = self.iota  

        #Calculating the discounted profits
        discounted_profits = self.R**(-np.arange(len(kappa_series))) * (profit_series - adjustment_cost_series) 

        #Calculating the ex post value of the salon giev the policy
        h_policy = np.sum(discounted_profits)  
        return h_policy
    
class HairSalonDynamicModel_Alternative_Policy:
    def __init__(self, eta=0.5, w=1.0, rho=0.90, iota=0.01, sigma_epsilon=0.10, R=1.01**(1/12), pricing_policy=None):
        """
        Initializing the HairSalonDynamicModel_Alternative_Policy class with the given parameters and pricing policy.
        
        """
        self.eta = eta
        self.w = w
        self.rho = rho
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R
        self.pricing_policy = pricing_policy

    def calculate_profit(self, kappa, ell=None):
        if ell is None:
            ell = ((1 - self.eta) * kappa / self.w) ** (1 / self.eta)
        profit = kappa * ell ** (1 - self.eta) - self.w * ell
        return profit

import numpy as np

class HairSalonDynamicModel_Alternative_Policy:
    def __init__(self, eta=0.5, w=1.0, rho=0.90, iota=0.01, sigma_epsilon=0.10, R=1.01**(1/12), pricing_policy=None):
        """
        Initializing the HairSalonDynamicModel_Alternative_Policy class with the given parameters and pricing policy.
        
        """
        self.eta = eta
        self.w = w
        self.rho = rho
        self.iota = iota
        self.sigma_epsilon = sigma_epsilon
        self.R = R
        self.pricing_policy = pricing_policy

    def calculate_profit(self, kappa, ell=None):
        if ell is None:
            ell = ((1 - self.eta) * kappa / self.w) ** (1 / self.eta)
        profit = kappa * ell ** (1 - self.eta) - self.w * ell
        return profit

    def calculate_h(self, epsilon_series):
        """
        Calculating the ex post value of the salon based on the epsilon series and alternative pricing policy.
        
        """
        #Creating a function to define the peak hours (peak hours = 9 AM to 5 PM, Monday to Friday )

        def is_peak_hour(t):
            if t.weekday() < 5 and t.hour >= 9 and t.hour < 17:
                return True
            else:
                return False
        
        #Creating a function to define the weekend (weekend = Saturday and Sunday)

        def is_weekend(t):
            if t.weekday() >= 5:
                return True
            else:
                return False
        
        #Calculating kappa_t using AR(1)
        kappa_series = np.exp(self.rho * np.concatenate(([0], epsilon_series[:-1])))

        #Calculating ell_t using the policy
        ell_series = ((1 - self.eta) * kappa_series / self.w) ** (1 / self.eta)

        #Initializing an array to store profits for each time step
        profit_series = np.zeros_like(kappa_series)

        #Initializing an array to store adjustment costs for each time step
        adjustment_cost_series = np.zeros_like(kappa_series)

        #Generate datetime objects for each time step
        time_steps = pd.date_range(start='2023-01-01', periods=len(kappa_series), freq='D')

        for t in range(len(kappa_series)):
            current_time_step = time_steps[t]

            #Applying alternative pricing policy if available
            if self.pricing_policy:
                pricing_factor = 1.0
                for condition, factor in self.pricing_policy.items():
                    if condition == 'peak_hours' and is_peak_hour(current_time_step):
                        pricing_factor *= factor
                    elif condition == 'weekends' and is_weekend(current_time_step):
                        pricing_factor *= factor
                kappa_series[t] *= pricing_factor

            #Calculating profit for each time step
            profit_series[t] = self.calculate_profit(kappa_series[t], ell_series[t])

    
            #Checking if ell changes from the time step before
            if t > 0 and ell_series[t] != ell_series[t - 1]:
                #Assigning adjustment cost if there is a change in ell
                adjustment_cost_series[t] = self.iota

        #Calculating discounted profits using the discount factor R
        discounted_profits = self.R ** (-np.arange(len(kappa_series))) * (profit_series - adjustment_cost_series)

        #Summing the discounted profits to get the ex post value of the salon
        h = np.sum(discounted_profits)
        return h








