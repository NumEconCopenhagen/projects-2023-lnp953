import numpy as np

class HairSalonModel:
    def __init__(self, eta=0.5, w=1.0):
        """
        Initializes the HairSalonModel class with the given parameters.
        
        Parameters:
        - eta: Elasticity of demand (Baseline: 0.5)
        - w: Wage for each hairdresser (Baseline: 1.0)
        """
        self.eta = eta
        self.w = w

    def calculate_profit(self, kappa, ell=None):
        """
        Calculates the profit based on the given kappa and ell values.
        
        Parameters:
        - kappa: Demand shock
        - ell: Number of hairdressers producing haircuts (Baseline: None)
        
        Returns:
        - profit: Calculated profit
        """
        if ell is None:
            ell = ((1 - self.eta) * kappa / self.w) ** (1 / self.eta)
        
        profit = kappa * ell ** (1 - self.eta) - self.w * ell
        
        return profit


class HairSalonDynamicModel:
    def __init__(self, eta=0.5, w=1.0, rho=0.90, iota=0.01, sigma_epsilon=0.10, R=1.01**(1/12)):
        """
        Initializes the HairSalonDynamicModel class with the given parameters.
        
        Parameters:
        - eta: Elasticity of demand (Baseline: 0.5)
        - w: Wage for each hairdresser (Baseline: 1.0)
        - rho: Autoregressive coefficient (Baseline: 0.90)
        - iota: Fixed adjustment cost (Baseline: 0.01)
        - sigma_epsilon: Standard deviation of the demand shock (Baseline: 0.10)
        - R: Discount factor (Baseline: (1+0.01)**(1/12))
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
        kappa_series = np.exp(self.rho * np.concatenate(([0], epsilon_series[:-1])))
        ell_series = ((1 - self.eta) * kappa_series / self.w) ** (1 / self.eta)
        profit_series = np.zeros_like(kappa_series)
        adjustment_cost_series = np.zeros_like(kappa_series)

        for t in range(len(kappa_series)):
            profit_series[t] = self.calculate_profit(kappa_series[t], ell_series[t])
            if t > 0 and ell_series[t] != ell_series[t-1]:
                adjustment_cost_series[t] = self.iota

        discounted_profits = self.R**(-np.arange(len(kappa_series))) * (profit_series - adjustment_cost_series)
        h = np.sum(discounted_profits)
        return h
