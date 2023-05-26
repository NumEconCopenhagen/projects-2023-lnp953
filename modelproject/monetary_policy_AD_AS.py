import numpy as np
import pandas as pd
import sympy as sm


class monetary_policy_intervention:
    def __init__(self):
        #Defining the symbols using sympy
        self.y, self.pi, self.g, self.b, self.alpha1, self.alpha2, self.h, self.s, self.y_bar, self.pi_star = sm.symbols(
            'y pi g b alpha1 alpha2 h s y_bar pi_star')

        #Defining the AD-curve
        z = self.alpha1 / (1 + self.alpha2 * self.b) * (self.g - self.y_bar)
        self.alpha = self.alpha2 * self.h / (1 + self.alpha2 * self.b)
        self.AD = self.y - self.y_bar - self.alpha * (self.pi - self.pi_star) - z

        #Defining the AS-curve
        self.gamma = sm.symbols('gamma')
        self.AS = self.pi - self.pi_star + self.gamma * (self.y - self.y_bar) + self.s

        #Steady state equation
        self.steady_state_eq = sm.solve(sm.Eq(self.AD, self.AS), self.y)[0]

        #Using lambdify to define the SS function
        self.ss_func = sm.lambdify(
            (self.pi, self.g, self.b, self.alpha1, self.alpha2, self.h, self.s, self.y_bar, self.pi_star, self.gamma),
            self.steady_state_eq)

    def ad(self, y, pi, pi_star, alpha, alpha_1, alpha_2, b, g, g_bar, h):
        #Defining the AD-curve equation
        z = alpha_1 / (1 + alpha_2 * b) * (g - g_bar)
        return y - self.y_bar - z + alpha * (pi - pi_star)

    def as_curve(self, y, pi, pi_1, gamma, s):
        #Defining the AS-curve equation
        return pi - pi_1 - gamma * (y - self.y_bar) - s

    def analyze_policy_intervention(self, interest_rate):
        #Defining the parameter values
        self.y_bar = 100
        self.pi_star = 2
        self.alpha_1 = 1
        self.alpha_2 = 1
        self.b = 1
        self.g = 1
        self.g_bar = 1
        self.h = 1
        self.alpha = self.alpha_2 * self.h / (1 + self.alpha_2 * self.b)
        self.gamma = 0.5

        #Assuming no supply shocks 
        self.s = 0

        #Calculating steady state before policy intervention
        steady_state_before = self.ss_func(self.pi_star, self.g, self.b, self.alpha_1, self.alpha_2, self.h, self.s,
                                           self.y_bar, self.pi_star, self.gamma)

        #Applying the policy intervention (decrease in interest rate)
        self.s = interest_rate

        #Calculating steady state after policy intervention
        steady_state_after = self.ss_func(self.pi_star, self.g, self.b, self.alpha_1, self.alpha_2, self.h, self.s,
                                          self.y_bar, self.pi_star, self.gamma)

        return steady_state_before, steady_state_after


