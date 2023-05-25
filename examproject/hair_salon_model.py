
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
            #If there is no value for ell, the code calculates it using the formula: ell = ((1 - eta) * kappa / w) ** (1 / eta)
            #The calculate value for ell is then assigned to the "ell" variable, which is then used in the profit calculation
            ell = ((1 - self.eta) * kappa / self.w) ** (1 / self.eta)
        
        #Calculating the profit using the formula: profit = kappa * ell ** (1 - eta) - w * ell
        profit = kappa * ell ** (1 - self.eta) - self.w * ell
        
        return profit
