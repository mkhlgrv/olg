import pandas as pd
from scipy.optimize import fsolve
import numpy as np
import itertools
import copy
from cyipopt import minimize_ipopt
import warnings
import matplotlib.pyplot as plt
from jax.config import config
import pickle
import sys
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import numpy as np
from jax import jit, grad, jacfwd, jacrev

from guess_plot import *
from progress_bar import *
import default_value as default

def labor_income_vector(self, s, g, start, end):
    return (1-self.tau_I[start:end]) * (1-(self.tau_rho[start:end] + self.tau_Ins[start:end])/(1+self.tau_rho[start:end] + self.tau_Ins[start:end])) * self.epsilon[s,g,start:end] * self.w[start:end]

def inf_to_zero(x):
    x[~np.isfinite(x)] = 0
    return x

plt.rc('legend', fontsize=12) 



class OLG_model:
    def __init__(self,
                 G=default.G_MODEL,
                 T=default.STEADY_TIME,
                 N=default.N,
                 Pi=default.Pi,
                 MAX_TIME = default.MAX_TIME,
                 epsilon=default.epsilon,
                 
                 beta=default.beta,
                 phi=default.phi,
                 upsilon = default.upsilon,
                 iota = default.iota,
                 utility = "hybrid",
                 
                 r=default.r,
                 price_M=default.price_M,
                 price_E=default.price_E,
                 omega=default.omega,
                 
                 alpha=default.alpha,
                 delta=default.delta,
                 psi=default.psi,
                 A=default.A,
                 
                 tau_I=default.tau_I,
                 tau_Ins=default.tau_Ins,
                 tau_pi=default.tau_pi,
                 tau_VA=default.tau_VA,
                 tau_rho=default.tau_rho,
                 tau_O = default.tau_O,
                 tax_LS = default.tax_LS,
                 
                 tax_sensitivity = default.tax_sensitivity,
                 target_debt_to_gdp = default.target_debt_to_gdp,
                 
                 sigma=default.sigma,
                 rho=default.rho,# or rho_reform
                 
                 price_O = default.price_O,
                 psi_O = default.psi_O,
                 Y_O = default.Y_O,
                 
                 initial=default.initial,
                 eta =default.eta,
                 steady_max_iter=default.steady_max_iter,
                 max_iter=default.max_iter,
                 steady_guess=default.steady_guess):
        """
        OLG model
        :param G: number of generations, default is 110
        :param T: simulation time, default is 500
        :param N_female: population (G+T-1, T) matrix where N[g,t] is cohort size of generation g in period t
        :param N_male: population (G+T-1, T) matrix where N[g,t] is cohort size of generation g in period t
        :param epsilon_female: female cohort- and year-specific productivity
        :param epsilon_male: male cohort- and year-specific productivity
        :param rho_female: female cohort- and year-specific retirement rate
        :param rho_male: male cohort- and year-specific retirement rate
        :param gov_strategy: "adaptive_gov", "adaptive_sigma", "adaptive_tau_rho", "unbalanced"
        :param gov_retirement_strategy: "unbalanced" for unbalanced retirement budget, "fixed_tau_rho" or "fixed_sigma" to zero-deficit retirement system
        :param sigma: pension rate (pension point value) as share of wage for T periods
        :param Pi_female: survival probability
        :param Pi_male: survival probability
        :param Beq_female_to_female,Beq_female_to_male,Beq_male_to_female,Beq_male_to_male: bequest matrix
        :param r: rate of return for T periods
        :param price_M: import prices for T periods
        :param price_E: export prices for T periods
        :param tau_I: income tax for T periods
        :param tau_Ins: insurance tax for T periods
        :param tau_pi: profit tax for T periods
        :param tau_VA: value-added tax for T periods
        :param tau_rho: retirement tax for T periods
        :param beta: discount factor
        :param phi: elasticity of intertemporal substitution
        :param psi: ? investment cost
        :param omega: import share
        :param alpha: elasticity of capital
        :param delta: deprecation rate
        :param A_N: total factor productivity, non-export good
        :param A_E: total factor productivity, export goods
        """

        ## Exogenous variables and parameters
        # Demography
        self.G, self.T,self.N,self.epsilon,self.Pi = G, T,N,epsilon,Pi
        # Utility
        self.beta, self.phi, self.upsilon, self.iota, self.utility = beta, phi,upsilon, iota,  utility
        
        
        # Prices
        self.r, self.price_M, self.price_E, self.omega= r, price_M, price_E, omega
        # Taxation
        self.tau_I,  self.tau_pi, self.tau_VA,self.tau_Ins, self.tau_rho, self.tau_O, self.tax_LS =  tau_I, tau_pi,tau_VA,tau_Ins, tau_rho, tau_O, tax_LS
        self.tax_sensitivity, self.target_debt_to_gdp = tax_sensitivity, target_debt_to_gdp
        
        # Production
        self.psi, self.alpha, self.delta, self.A = psi, alpha, delta, A

        # Retirement
        self.rho, self.sigma = rho, sigma
        
        # Oil
        self.price_O = price_O
        self.psi_O = psi_O
        self.Y_O = Y_O

        self.initial = initial

        self.last_guess = None

        self.eta = eta
        self.steady_max_iter = steady_max_iter
        self.max_iter = max_iter
        self.max_time = MAX_TIME

        # Endogenous variable
        # Initial guess
        self.price_N = np.repeat(initial["price_N"], self.max_time)
        
        self.K = np.array([np.repeat(initial["K_N"], self.max_time),np.repeat(initial["K_E"], self.max_time)])

        self.L = np.array([np.repeat(initial["L_N"], self.max_time),np.repeat(initial["L_E"], self.max_time)])
        
        self.I = np.array([np.repeat(initial["I_N"], self.max_time),np.repeat(initial["I_E"], self.max_time)])
        
        
        self.Debt = np.repeat(initial["Debt"], self.max_time)
        
        self.lmbda = np.ones_like(self.K)

        self.w = np.ones(self.max_time)

        self.price = np.ones(self.max_time)


        
        self.k = np.ones_like(self.K)
        self.k_guess = self.k.copy()

        self.k[0,0] = self.K[0,0]/(self.L[:,0].sum()*self.A[0,0])
        self.k[0,1] = (self.K[0,0] * (1-self.delta) + self.I[0,0])/(self.L[:,1].sum() *self.A[0,1])
        
        self.k[1,0] = self.K[1,0] /(self.L[:,0].sum()*self.A[1,0])
        self.k[1,1] = (self.K[1,0] * (1-self.delta) + self.I[1,0])/(self.L[:,1].sum()*self.A[1,1])
        
        self.i = np.ones_like(self.I)
        self.i[0,0] = self.I[0,0]/self.K[0,0]
        self.i[1,0] = self.I[1,0]/self.K[1,0]
        self.i_guess = self.i.copy()        
        self.lmbda_to_price = np.ones_like(self.lmbda)

        self.l_demand = np.ones_like(self.L)
        self.l_demand[0] = self.L[0, 0]/self.L[:,0].sum()
        self.l_demand[1] = self.L[1, 0]/self.L[:,0].sum()
        
        self.gov_adaptation_time = None
        


        self.c =  np.array([[[0.5 if ((g >= t) and (g<=self.G+t-1)) else 0 for t in range(self.max_time)] 
                             for g in range(self.max_time)] for _ in range(2)])
        
        self.a = np.array([[[0.1 if ((g >= t) and (g<=self.G+t-1)) else 0 for t in range(self.max_time)] for g in range(self.max_time)] for _ in range(2)])
        
        self.gamma = np.array([[[0.9 if ((g >= t) and (g<=self.G+t-1)) else 0 for t in range(self.max_time)] for g in range(self.max_time)] for _ in range(2)])
        
        self.l = np.array([[[1. if ((g >= t) and (g<=self.G+t-1)) else 0 for t in range(self.max_time)] for g in range(self.max_time)] for _ in range(2)])

        self.Consumption = np.array([np.sum([self.c[s,g,t]*self.N[s,g,t] 
                                        for g in range(self.max_time) 
                                        for s in range(2)]) for t in range(self.max_time)])
        
        self.Labor = np.array([np.sum([self.l[s,g,self.T]*self.N[s,g,self.T]*self.epsilon[s,g,self.T] 
                                  for g in range(self.max_time) 
                                  for s in range(2)]) for t in range(self.max_time)])
        
        self.Labor[0] = self.L[:,0].sum()

        self.Assets =  np.array([np.sum([self.a[s,g,t]*self.N[s,g,t] 
                                         for g in range(self.max_time) 
                                         for s in range(2)]) 
                                 for t in range(self.max_time)])
        
        self.steady_state = steady_guess
        self.steady_state_path = []
        
        self.A_growth = self.A[0,self.T]/self.A[0, self.T-1]
        
        self.N_growth = np.sum([self.N[s,g,self.T]*self.epsilon[s,g,self.T] 
                                for g in range(self.T, self.G+self.T) 
                                for s in range(2)])/\
                   np.sum([self.N[s,g,self.T-1]*self.epsilon[s,g,self.T-1] 
                           for g in range(self.T-1, self.G+self.T-1) 
                           for s in range(2)])
        
        self.potential_growth = np.concatenate(([1.],
                                               (self.N[:,:,1:self.max_time]*self.epsilon[:,:,1:self.max_time]).sum(axis=(0,1))*self.A[0,1:]/\
                                               ((self.N[:,:,0]*self.epsilon[:,:,0]).sum(axis=(0,1)))))
        
        self.working_households = np.array([(self.N[:, :self.max_time, t]*self.l[:, :, t]*self.w[t]).sum(axis=(0,1)) for t in range(self.max_time)])
        
        self.history = {t:[] for t in range(self.max_time)}

    @property
    def Y(self):
        return np.array([self.K[S]**self.alpha * (self.L[S]*self.A[S])**(1-self.alpha) for S in range(2)])
    
    @property
    def lmbda_to_price_steady(self):
        return 1/((1 - self.psi/2 * (self.A_growth*self.N_growth -1)**2 - self.psi*self.A_growth*self.N_growth * (self.A_growth*self.N_growth -1))+self.psi/(1+self.r[self.T+1]) * (self.A_growth*self.N_growth -1)* (self.A_growth*self.N_growth)**2)

    @property
    def i_steady(self):
        return (self.A_growth*self.N_growth-1+self.delta) / (1-self.psi/2*(self.A_growth*self.N_growth -1)**2)

    @property 
    def D(self):
        return self.Consumption+self.I.sum(axis=0) + self.Y_O * self.psi_O

    @property 
    def M(self):
        return self.omega * self.D * self.price / self.price_M
    
    @property 
    def tax_LS_sum(self):
        return self.tax_LS*self.N.sum(axis=(0,1))

    @property
    def Gov(self):
        return self.initial["Gov"]*self.potential_growth
    


    @property
    def GDP(self):
        return self.Y_O+self.Y.sum(axis=0)
    @property
    def Gov_to_GDP(self):
        return self.Gov/self.GDP

    @property
    def VA_sum(self):
        return self.tau_VA*self.price*(self.Consumption)
    @property
    def O_sum(self):
        return self.tau_O*(self.price_O-self.psi_O)* self.Y_O

    @property
    def I_sum(self):
        return self.Labor * self.w *(1-(self.tau_rho + self.tau_Ins)/(1+self.tau_rho + self.tau_Ins)) *\
                    self.tau_I
    @property
    def Ins_sum(self):
        return self.Labor * self.w *self.tau_Ins/(1+self.tau_rho + self.tau_Ins)
    @property
    def Rho_sum(self):
        return self.tau_rho/(1+self.tau_rho + self.tau_Ins) * self.Labor * self.w
    @property 
    def Pi_sum(self):
        return self.tau_pi * ([self.price_N,self.price_E]  * self.K**self.alpha *\
                                           (self.L*self.A)**(1-self.alpha) -
                                 self.w*self.L - self.delta * self.price*self.K).sum(axis=0)
    @property
    def Gov_Income(self):
        return self.VA_sum + self.I_sum+self.Ins_sum+self.Rho_sum+self.Pi_sum+self.O_sum
    
    
    @property
    def Rho_Outcome(self):
        return self.Labor/self.working_households * self.w * (self.sigma[0]* (self.rho[0]*self.N[0]).sum(axis=(0))+\
                self.sigma[1] * (self.rho[1]*self.N[1]).sum(axis=(0)))
    
    @property
    def Gov_Outcome(self):
        return self.price_N*self.Gov +self.Rho_Outcome+\
    self.r*np.concatenate(([self.initial["Debt"]],self.Debt[:-1]))+\
        self.tax_LS_sum
    
    @property
    def Deficit(self):
        return self.Gov_Outcome - self.Gov_Income
    @property
    def Deficit_rho(self):
        return self.Rho_Outcome-self.Rho_sum
    @property 
    def Deficit_rho_to_GDP(self):
        return self.Deficit_rho/self.price/self.GDP
    
    @property
    def Debt_to_GDP(self):
        return self.Debt/self.price/self.GDP
    
    @property
    def Deficit(self):
        return self.Gov_Outcome - self.Gov_Income
    
    @property
    def Deficit_to_GDP(self):
        return self.Deficit/self.price/self.GDP
        
        
#     def copy(self,model):

#         self.w = model.w
#         self.price= model.price
#         self.price_N = model.price_N

#         self.k = model.k
#         self.i = model.i

#         self.l_demand = model.l_demand

#         self.lmbda_to_price =  model.lmbda_to_price

#         self.K = model.K
#         self.I = model.I
#         self.L = model.L
#         self.lmbda = model.lmbda
#         self.Consumption = model.Consumption
#         self.Labor = model.Labor
#         self.Assets = model.Assets
#         self.c = model.c
#         self.l = model.l
#         self.a = model.a
#         self.sigma = model.sigma
#         self.rho = model.rho
        
#         self.tau_rho = model.tau_rho
#         self.tau_VA = model.tau_VA
#         self.tau_I = model.tau_I
#         self.Debt = model.Debt
#         self.steady_state = model.steady
        
        

        
    def update_working_households(self, t):
        self.working_households[t] = (self.N[:, :self.max_time, t]*self.l[:, :, t]).sum(axis=(0,1))
        
    def update_government(self, t,i=0):
            
        def reaction_function(self, t):
            if t>0:
                debt_deviation = (self.Debt_to_GDP[t-1] - self.target_debt_to_gdp)
                
                I = max(0, min(self.tau_I[0]+self.tax_sensitivity["I"]*debt_deviation + self.tax_sensitivity["I_lag"]*self.tau_I[t-1], 0.8))
                VA = max(0.2, min((1-self.tax_sensitivity["VA_lag"])*self.tau_VA[0]+\
                                 self.tax_sensitivity["VA"]*debt_deviation +\
                                 self.tax_sensitivity["VA_lag"]*self.tau_VA[t-1], 0.8))
                
                return I, VA
            else:
                return self.tau_I[t], self.tau_VA[t]
            
            
        self.tau_I[t], self.tau_VA[t] = reaction_function(self, t)
                
        if t == 0:
            self.Debt[t] = self.initial["Debt"]+self.Deficit[t]
        else:
            self.Debt[t] = self.Debt[t-1]+self.Deficit[t]

            
    def household(self, s, g, t, t_0=None):
        def cumulative_rate_of_return(self,start, end):
            if start <= end:
                return np.prod(1+self.r[start:end])
            else:
                return 1



        def get_initial_consumption(self, s, g, t_0):

            if self.utility == "exogenous_labor": # нулевая эластичность труда, выбирается только потребление. labor = phi[s]
                numerator = \
                    (
                        self.a[s,g,t_0-1]* cumulative_rate_of_return(self, t_0, g+1)+\
                        np.sum(
                            np.array([cumulative_rate_of_return(self,start, g+1) for start in range(t_0+1, g+2)])*\
                            (
                                labor_income_vector(self, s, g, t_0, g+1)*self.phi[s]+\
                                self.tax_LS[t_0:( g+1)]+\
                                self.rho[s,g,t_0:(g+1)]*self.sigma[s, t_0:(g+1)]*self.w[t_0:(g+1)]*(self.Labor[t_0:(g+1)]/self.working_households[t_0:(g+1)])
                            )
                        )
                    )
                denominator = \
                    (
                        cumulative_rate_of_return(self,t_0+1, g+1)*
                        np.sum(
                            (1+self.tau_VA[t_0:(g+1)])*(self.price[t_0:(g+1)])*\
                            self.beta**np.array([i-t_0 for i in range(t_0, g+1)])*\
                            self.Pi[s,g,t_0:(g+1)]/self.Pi[s,g,t_0] *\
                            (1+self.tau_VA[t_0])*(self.price[t_0])/\
                            (
                               (1+self.tau_VA[t_0:(g+1)])*(self.price[t_0:(g+1)])
                            )
                        )
                    )
                return numerator / denominator
            
            
            elif self.utility == "cobb_douglas": # единичная эластичность, труд сильно реагирует на зп
                return (self.a[s,g,t_0-1]* cumulative_rate_of_return(self, t_0, g+1)+
                    np.sum(np.array([cumulative_rate_of_return(self,start, g+1) for start in range(t_0+1, g+2)])*
                    (labor_income_vector(self, s, g, t_0, g+1)+\
                    self.tax_LS[t_0:( g+1)]+
                    self.rho[s,g,t_0:(g+1)]*self.sigma[s, t_0:(g+1)]*self.w[t_0:(g+1)]*(self.Labor[t_0:(g+1)]/self.working_households[t_0:(g+1)])
                    )))/\
                    (1/(1-self.phi)*np.sum(
                    np.array([cumulative_rate_of_return(self,start, g+1) for start in range(t_0+1, g+2)])*
                    (1+self.tau_VA[t_0:(g+1)])*(self.price[t_0:(g+1)])*
                    np.array([cumulative_rate_of_return(self,t_0+1, end) for end in range(t_0+1, g+2)])*
                    self.beta**np.array([i-t_0 for i in range(t_0, g+1)])*self.Pi[s,g,t_0:(g+1)]/self.Pi[s,g,t_0] * (1+self.tau_VA[t_0])*(self.price[t_0])/((1+self.tau_VA[t_0:(g+1)])*(self.price[t_0:(g+1)]))*\
                    np.sign(self.epsilon[s,g,t_0:g+1])
                    )+\
                    np.sum(
                    np.array([cumulative_rate_of_return(self,start, g+1) for start in range(t_0+1, g+2)])*
                    (1+self.tau_VA[t_0:(g+1)])*(self.price[t_0:(g+1)])*
                    np.array([cumulative_rate_of_return(self,t_0+1, end) for end in range(t_0+1, g+2)])*
                    self.beta**np.array([i-t_0 for i in range(t_0, g+1)])*self.Pi[s,g,t_0:(g+1)]/self.Pi[s,g,t_0] * (1+self.tau_VA[t_0])*(self.price[t_0])/((1+self.tau_VA[t_0:(g+1)])*(self.price[t_0:(g+1)]))*\
                    (1-np.sign(self.epsilon[s,g,t_0:g+1]))
                    ))
            

            elif self.utility == "hybrid": # гибридная функция с эластичностью < 1
                def life_time_budget_constraint_equation(x):
                    return - (self.a[s, g, t_0 - 1] * cumulative_rate_of_return(self, t_0, g + 1) +
                         np.sum(
                             np.array([cumulative_rate_of_return(self, start, g + 1) for start in range(t_0 + 1, g + 2)]) *
                             (self.tax_LS[t_0:(g + 1)] +
                              self.rho[s, g, t_0:(g + 1)] * self.sigma[s, t_0:(g + 1)] * self.w[t_0:(g + 1)]*\
                              (self.Labor[t_0:(g+1)]/self.working_households[t_0:(g+1)])
                              ))) +\
                        cumulative_rate_of_return(self, t_0+1, g+1) * \
                            np.sum(
                                self.beta ** np.array([i - t_0 for i in range(t_0, g + 1)]) * self.Pi[s, g, t_0:(g + 1)] /
                                self.Pi[s, g, t_0] * (1 + self.tau_VA[t_0]) * (self.price[t_0])
                            )*  x- \
                               np.sum(
                                   np.array(
                                       [cumulative_rate_of_return(self, start, g + 1) for start in range(t_0 + 1, g + 2)]) *
                                   labor_income_vector(self, s, g, t_0, g + 1)**((self.upsilon+1)/self.upsilon)*
                                   (
                                           self.iota[s] * np.array([cumulative_rate_of_return(self, t_0 + 1, end) for end in
                                                     range(t_0 + 1, g + 2)]) *
                                           self.beta ** np.array([i - t_0 for i in range(t_0, g + 1)]) * self.Pi[s, g,
                                                                                                         t_0:(g + 1)] /
                                           self.Pi[s, g, t_0] * (1 + self.tau_VA[t_0]) * (self.price[t_0])

                                   )**(-1/self.upsilon)
                               ) * x**(-1/self.upsilon)
                return fsolve(life_time_budget_constraint_equation, 0.5)


        bequest = 0.
        if t_0 is None:
            t_0 = max(g-self.G+1,0)
        if g >= t >=g-self.G+1:
            if t==t_0:
                consumption = get_initial_consumption(self, s, g, t)
            else:
                t_0 = max(g-self.G+1,t_0)
                if (self.utility == "exogenous_labor") or (self.utility == "cobb_douglas") or (self.utility == "hybrid"): # нулевая эластичность труда, выбирается только потребление
                # или единичная эластичность, труд сильно реагирует на зп
                
                    consumption = self.c[s, g, t_0]*\
                              (cumulative_rate_of_return(self, t_0+1, t+1)*
                               self.beta**(t-t_0)*self.Pi[s,g,t]/self.Pi[s,g,t_0] * (1+self.tau_VA[t_0])*\
                               self.price[t_0]/((1+self.tau_VA[t])*self.price[t])
                               )
                

            if self.epsilon[s,g,t] == 0:
                labor = 0
            else:
                if self.utility == "exogenous_labor": # нулевая эластичность труда, выбирается только потребление, труд всегда фикс
                    labor = self.phi[s]
                    
                elif self.utility == "cobb_douglas": # единичная эластичность, труд сильно реагирует на зп
                
                    labor = 1- consumption/((1-self.phi)/(self.phi)*(1/((1+self.tau_VA[t])*self.price[t]))*labor_income_vector(self, s, g, t, t+1)[0])
                    
                    

                elif self.utility == "hybrid": # гибридная функция
                    labor = (labor_income_vector(self, s, g, t, t + 1) / (consumption * self.iota[s] * (1+self.tau_VA[t])*self.price[t]))**(1/self.upsilon)

            assets = self.tax_LS[t]+labor_income_vector(self, s, g, t, t+1)[0]*labor+self.rho[s,g,t]*self.sigma[s, t]*self.w[t]*(self.Labor[t]/self.working_households[t]) - consumption*(1+self.tau_VA[t])*self.price[t]+(self.a[s,g, t-1]+bequest)*(1+self.r[t]*(1-self.tau_I[t]))
            return consumption, labor, assets
        else:
            return np.array([0,0,0])


    
    def update_steady_state(self):

        w_steady, price_steady, price_N_steady = self.steady_state[3:6]

                

        for t in range(self.T-self.G + 1, self.max_time):
            self.w[t] = w_steady* self.A_growth**(t - self.T)
            self.price[t] = price_steady
            self.price_N[t] = price_N_steady

        for s in range(2):
            initial_household = self.household(s, self.G+self.T-1, self.T,self.T) # первое родившееся поколение в период self.T
            self.c[s, self.G+self.T-1, self.T],self.l[s, self.G+self.T-1,self.T],self.a[s, self.G+self.T-1,self.T] = initial_household
            for t in range(self.T,self.G+self.T):
                self.c[s, self.G+self.T-1, t],self.l[s, self.G+self.T-1,t],self.a[s, self.G+self.T-1,t] = self.household(s, self.G+self.T-1, t,t)
            for g in range(self.G+self.T-2, self.T-1, -1): # все поколения, живущие в периоде self.T
                
                self.c[s, g, (g-self.G + 1):(g+1)]  = self.c[s, self.G+self.T-1, (self.T):(self.G+self.T)] \
                * self.A_growth**(g-self.G + 1 - self.T)
                self.l[s, g, (g-self.G + 1):(g+1)]  = self.l[s, self.G+self.T-1, (self.T):(self.G+self.T)]
                self.a[s, g,(g-self.G + 1):(g+1)]  =self.a[s, self.G+self.T-1, (self.T):(self.G+self.T)]* self.A_growth**(g-self.G + 1 - self.T)
                
        Consumption = np.sum([self.c[s,g,self.T]*self.N[s,g,self.T] 
                              for g in range(self.T, self.G+self.T) 
                              for s in range(2)])

        Labor = np.sum([self.l[s,g,self.T]*self.N[s,g,self.T]*self.epsilon[s,g,self.T] for g in range(self.T, self.G+self.T) for s in range(2)])
        

        Assets =  np.sum([self.a[s,g,self.T]*self.N[s,g,self.T] for g in range(self.T, self.G+self.T) for s in range(2)])
        
        if len(self.steady_state_path)==0:
            
            self.steady_state[-3:] = self.eta*np.array([Consumption,  Labor, Assets]) + (1-self.eta)*self.steady_state[-3:]
            gov = 0.35
            oil_costs = 0.
        else:
            # self.steady_state[-3:] = self.eta * np.array([Consumption,  Labor, Assets]) + (1-self.eta)*self.steady_state[-3:]
            self.steady_state[-3:] = np.array([Consumption,  Labor, Assets])
        
            self.K[0,self.T] = self.steady_state[0]* Labor * self.A[0,self.T] 
            self.K[1,self.T] = self.steady_state[2]* Labor * self.A[0,self.T]
            
            self.I[0,self.T] = self.i_steady * self.K[0,self.T]
            self.I[1,self.T] = self.i_steady* self.K[1,self.T]
            
            self.L[0,self.T] = self.steady_state[1] * Labor 
            self.L[1,self.T] = (1-self.steady_state[1]) * Labor
            
 

            gov = self.Gov[self.T]/(self.A[0,self.T]*self.steady_state[-2])
            oil_costs = self.psi_O[self.T] * self.Y_O[self.T]/(self.A[0,self.T]*self.steady_state[-2])
            

        z_guess = self.steady_state[:6]
        
        def equilibrium(z, self=self, objective = True):
              
            system = [
                f"{1-self.alpha}*price_N_steady * (k_N_steady/l_N_demand)**{self.alpha}- w_steady/{self.A[0,self.T]}"
                      ,f"{1/(1+self.r[self.T+1])}*(({1-self.tau_pi[self.T+1]}) *{self.alpha}*price_N_steady* (k_N_steady/l_N_demand)**{self.alpha-1} +{self.tau_pi[self.T+1]} * {self.delta}*price_steady + {self.lmbda_to_price_steady}*price_steady * ({1-self.delta})) - {self.lmbda_to_price_steady}*price_steady"
                      ,f"{1-self.alpha}*{self.price_E[self.T]} * (k_E_steady/(1-l_N_demand)*{self.A[0,self.T]/self.A[1,self.T]} )**{self.alpha} - w_steady/{self.A[1,self.T]}"
                      ,f"{1/(1+self.r[self.T+1])}*({1-self.tau_pi[self.T+1]}*{self.alpha}* {self.price_E[self.T+1]}* (k_E_steady/(1-l_N_demand)*{self.A[0,self.T]/self.A[1,self.T]})**{self.alpha-1} +{self.tau_pi[self.T+1]} * {self.delta}*price_steady +{self.lmbda_to_price_steady}*price_steady * {1-self.delta}) - {self.lmbda_to_price_steady}*price_steady"
                      ,f"price_N_steady*((k_N_steady/l_N_demand)**{self.alpha} * l_N_demand -{gov})- (1-{self.omega}) * price_steady * ({self.steady_state[-3]/(self.A[0,self.T]*self.steady_state[-2])}+{self.i_steady}*(k_N_steady+k_E_steady) + {oil_costs})"
                      ,f"price_steady - {self.price_M[self.T]**self.omega}*price_N_steady**({1-self.omega})"
                      ]
            name_space = {label:value for label, value in zip(("self", "k_N_steady", "l_N_demand",
                                                               "k_E_steady", "w_steady",
                                                               "price_steady", "price_N_steady"),[self]+list(z))}

            if objective:
                sum_of_squares = "+".join([f"({equation})**2" for equation in system])
                F = eval(sum_of_squares,{},name_space)
            else:
                F = [eval(equation,{},name_space) for equation in system]
            return F
        

        
        obj_jit = jit(equilibrium)
        obj_grad = jit(jacfwd(obj_jit))
        obj_hess = jit(jacrev(jacfwd(obj_jit)))
        
        
        result = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=z_guess
                                , options = {"max_iter":self.steady_max_iter, "print_level":0,
                                             "check_derivatives_for_naninf":"yes"}
                                , tol=1e-10)
        
        
        if result["success"] or result["status"]==1:
            self.steady_state[:6] = self.eta*result["x"] + (1-self.eta)*z_guess
            self.steady_max_iter = 10000
            eq_res = equilibrium(self.steady_state[:6], self, False)
        else:
            if result["status"]==-1:
                self.steady_max_iter += 1000
            self.steady_state[:6] = z_guess
            eq_res = equilibrium(z_guess, self, False)
            
                
        self.steady_state_path.append((result, np.array(self.steady_state)
                               ))

    def create_guess(self, t_0=0, steady_start=None):
        
        k_N_steady, l_N_demand, k_E_steady, w_steady, price_steady, price_N_steady, Consumption_steady, Labor_steady, Assets_steady = self.steady_state
        if steady_start is None:
            steady_start = self.T-self.G

        
        for t in range(steady_start, self.max_time):
            
            self.w[t] = w_steady* self.A_growth**(t - self.T)
            self.price[t] = price_steady
            self.price_N[t] = price_N_steady
            
            self.k[0,t] = k_N_steady
            self.k[1,t] = k_E_steady
            
            self.i[0,t] = self.i_steady
            self.i[1,t] = self.i_steady 
            
            self.l_demand[0,t] = l_N_demand 
            self.l_demand[1,t] = 1- l_N_demand 
            
            self.lmbda_to_price[0,t] = self.lmbda_to_price_steady
            self.lmbda_to_price[1,t] = self.lmbda_to_price_steady
            
            
            
            self.K[0,t] = k_N_steady* Labor_steady * self.A[0,self.T] *\
                (self.A_growth*self.N_growth)**(t - self.T)
            self.K[1,t] = k_E_steady* Labor_steady * self.A[0,self.T] *\
                (self.A_growth*self.N_growth)**(t - self.T)
            
            self.I[0,t] = self.i_steady * self.K[0,t]
            self.I[1,t] = self.i_steady * self.K[1,t]
            
            self.L[0,t] = l_N_demand * Labor_steady *(self.N_growth)**(t - self.T)
            self.L[1,t] = (1-l_N_demand) * Labor_steady *(self.N_growth)**(t - self.T)
            
            self.lmbda[0,t] = self.lmbda_to_price_steady * price_steady
            self.lmbda[1,t] = self.lmbda_to_price_steady * price_steady
            
            self.Consumption[t] = Consumption_steady*(self.A_growth*self.N_growth)**(t - self.T)
            self.Labor[t] = Labor_steady*(self.N_growth)**(t - self.T)
            self.Assets[t] =  Assets_steady*(self.A_growth*self.N_growth)**(t - self.T)
            
            
        
        
        
        if t_0==1:
            # self.k[0,(t_0+1):(steady_start+1)] = np.linspace(self.k[0,0], self.k[0,steady_start],\
            #                                       steady_start-t_0,endpoint=False)
            # self.k[1,(t_0+1):(steady_start+1)] = np.linspace(self.k[1,0], self.k[1,steady_start],\
            #                                           steady_start-t_0,endpoint=False)

            #self.i[0,t_0:steady_start] = np.linspace(self.i[0,0], self.i[0,steady_start],\
            #                                         steady_start-t_0,endpoint=False)
            #self.i[1,t_0:steady_start] = np.linspace(self.i[1,0], self.i[1,steady_start],\
            #                                          steady_start-t_0,endpoint=False)

            self.l_demand[0,t_0:steady_start] = np.linspace(self.l_demand[0,0], self.l_demand[0,steady_start],\
                                                      steady_start-t_0,endpoint=False)
            self.l_demand[1,t_0:steady_start] = np.linspace(self.l_demand[1,0], self.l_demand[1,steady_start],\
                                                      steady_start-t_0,endpoint=False)


            self.evaluate_initial_state()
            self.w[:steady_start] = np.linspace(self.w[0], self.w[steady_start], steady_start,endpoint=False)
            self.price_N[:steady_start] = np.linspace(self.price_N[0], self.price_N[steady_start], steady_start,endpoint=False)
            self.price[:steady_start] = np.linspace(self.price[0], self.price_N[steady_start], steady_start,endpoint=False)
        else:
            # self.k[0,(t_0+1):(steady_start+1)] = np.linspace(self.k[0,t_0], self.k[0,steady_start],\
            #                                       steady_start-t_0,endpoint=False)
            # self.k[1,(t_0+1):(steady_start+1)] = np.linspace(self.k[1,t_0], self.k[1,steady_start],\
            #                                           steady_start-t_0,endpoint=False)

            #self.i[0,t_0:steady_start] = np.linspace(self.i[0,t_0], self.i[0,steady_start],\
            #                                          steady_start-t_0,endpoint=False)
            #self.i[1,t_0:steady_start] = np.linspace(self.i[1,t_0], self.i[1,steady_start],\
            #                                          steady_start-t_0,endpoint=False)

            self.l_demand[0,t_0:steady_start] = np.linspace(self.l_demand[0,t_0], self.l_demand[0,steady_start],\
                                                      steady_start-t_0,endpoint=False)
            self.l_demand[1,t_0:steady_start] = np.linspace(self.l_demand[1,t_0], self.l_demand[1,steady_start],\
                                                      steady_start-t_0,endpoint=False)
            
            self.w[t_0:steady_start] = np.linspace(self.w[t_0], self.w[steady_start], steady_start-t_0,endpoint=False)
            self.price_N[t_0:steady_start] = np.linspace(self.price_N[t_0], self.price_N[steady_start], steady_start-t_0,endpoint=False)
            self.price[t_0:steady_start] = np.linspace(self.price[t_0], self.price[steady_start], steady_start-t_0,endpoint=False)

     
    
    
        self.lmbda_to_price[0,t_0:steady_start] = self.lmbda_to_price_steady
        self.lmbda_to_price[1,t_0:steady_start] = self.lmbda_to_price_steady
        self.lmbda[0,t_0:steady_start] = self.price[t_0:steady_start]*self.lmbda_to_price_steady
        self.lmbda[1,t_0:steady_start] = self.price[t_0:steady_start]*self.lmbda_to_price_steady

        for t in range(t_0,self.T):
            for s in range(2):
                for g in range(t, self.G+t):
                    self.c[s, g, t], self.l[s,g,t], self.a[s,g,t] = self.household(s,g,t, t)
                    
        self.Consumption[t_0:steady_start] = np.array([np.sum(self.c[:,:,t]*self.N[:,:self.c.shape[1],t]) for t in range(t_0,steady_start)])
        self.Labor[t_0:steady_start] = np.array([np.sum(self.l[:,:,t]*self.N[:,:self.l.shape[1],t]*self.epsilon[:,:self.l.shape[1],t]) for t in range(t_0,steady_start)])
        self.Assets[t_0:steady_start] = np.array([np.sum(self.a[:,:,t]*self.N[:,:self.a.shape[1],t]) for t in range(t_0,steady_start)])
        
        
        # if t_0==1:
#             self.k[0,1] = (self.K[0,0] * (1-self.delta) + self.I[0,0])/(self.Labor[1]*self.A[0,1])
#             self.k[1,1] = (self.K[1,0] * (1-self.delta) + self.I[1,0])/(self.Labor[1]*self.A[0,1])
            
#             self.k[0,(t_0+1):(steady_start+1)] = np.linspace(self.k[0,1], self.k[0,steady_start],\
#                                                   steady_start-t_0,endpoint=False)
#             self.k[1,(t_0+1):(steady_start+1)] = np.linspace(self.k[1,1], self.k[1,steady_start],\
#                                                       steady_start-t_0,endpoint=False)
            
        # self.K[:,t_0:steady_start] = self.k[:,t_0:steady_start] * self.A[0,t_0:steady_start]*\
        #     np.array([self.Labor[t_0:steady_start] for _ in range(2)])                                                                                                               
        
        
        self.L[:,t_0:steady_start] = self.l_demand[:,t_0:steady_start] * np.array([self.Labor[t_0:steady_start] for _ in range(2)])
        self.I[0,t_0:steady_start] = np.linspace(self.I[0,0], self.I[0,steady_start],\
                                                      steady_start-t_0,endpoint=False)
        self.I[1,t_0:steady_start] = np.linspace(self.I[1,0], self.I[1,steady_start],\
                                                      steady_start-t_0,endpoint=False)


        for S in range(2):
            if t_0==1:
                self.K[S, t_0] = self.K[S, t_0-1] * (1-self.delta) + self.I[S, t_0-1]
            
            self.K[S,(t_0+1):(steady_start+1)] = np.linspace(self.K[S,t_0], self.K[S,steady_start],\
                                                  steady_start-t_0,endpoint=False)
            self.I[S,t_0:steady_start] = np.linspace(self.I[S,0], self.I[S,steady_start],\
                                                      steady_start-t_0,endpoint=False)
            self.i[S, :] = self.I[S, :] / self.K[S, :]
            self.k[S, (t_0+1):(steady_start+1)] = self.K[S, (t_0+1):(steady_start+1)] / (self.Labor[(t_0+1):(steady_start+1)]*self.A[S, (t_0+1):(steady_start+1)])
        
            
            
    def evaluate_initial_state(self):
        
        t=0
        self.w[t] = (1-self.alpha)*self.price_E[t] * (self.k[1, t]/\
                                                      (1-self.l_demand[0,t])*self.A[0,t]/self.A[1,t])**self.alpha* self.A[1, t] 
        self.price_N[t] = self.w[t]/self.A[0, t]/((1-self.alpha)  * \
                                                  (self.k[0, t]/self.l_demand[0,t])**self.alpha)
        
        self.price[t] =   (self.price_N[t]*((self.k[0,t]/self.l_demand[0,t])**self.alpha *self.l_demand[0,t]-self.Gov[t]/(self.A[0,t]*self.Labor[t])))/\
                             ((1-self.omega) *  (self.Consumption[t]/\
                                                 (self.A[0,t]*self.Labor[t]) +self.i[0,t]*self.k[0,t] + self.i[1,t]*self.k[1,t]))

    def update_a_initial(self):
        self.Assets[0] = self.steady_state[-1]/self.steady_state[-3] * self.Consumption[0]
        coef = np.sum(self.a[:,self.G+self.T-2:self.T-1:-1,self.T]*\
                      self.N[:,self.G+0-2::-1,0])/self.Assets[0]
        
        self.a[:,self.G-2::-1,0] = self.a[:,self.G+self.T-2:self.T-1:-1,self.T]/coef
        

    def update_household(self,t, t_0=None):              
        for s in range(2):
            for g in range(t, self.G+t):
                self.c[s, g, t],self.l[s, g, t],self.a[s, g,t]  = self.household(s,g,t,t)

        self.Consumption[t] = np.sum([self.c[s,g,t]*self.N[s,g,t] 
                                      for g in range(t, self.G+t) 
                                      for s in range(2)])
        self.Labor[t] = np.sum([self.l[s,g,t]*self.N[s,g,t]*self.epsilon[s,g,t] 
                                for g in range(t, self.G+t) 
                                for s in range(2)])
        self.update_working_households(t) 
        self.Assets[t] = np.sum([self.a[s,g,t]*self.N[s,g,t]
                                 for g in range(t, self.G+t) 
                                 for s in range(2)])
        
    def update_guess(self, t, t_0=1):
        
       # self.L[:,t_0:self.T] = self.l_demand[:,t_0:self.T]*self.Labor[t_0:self.T]

       # self.K[0,(t_0+1):(self.T+1)] = self.k[0,(t_0+1):(self.T+1)]*self.Labor[(t_0+1):(self.T+1)]*self.A[0,(t_0+1):(self.T+1)]
       # self.K[1,(t_0+1):(self.T+1)] = self.k[1,(t_0+1):(self.T+1)]*self.Labor[(t_0+1):(self.T+1)]*self.A[0,(t_0+1):(self.T+1)] # тут 0, а не 1!

       # self.I[:,t_0:self.T] = self.i[:,t_0:self.T]*self.K[:,t_0:self.T]

        

        z_guess = np.array([self.i[0,t],self.k[0,t+1],self.l_demand[0,t],
                       self.lmbda_to_price[0,t], self.i[1,t],self.k[1,t+1],
                       self.lmbda_to_price[1,t],
                       self.w[t], self.price_N[t], self.price[t]])


        def equilibrium(z, self=self, t = t):
            i_N, k_N, l_N_demand, lmbda_N_to_price,\
                i_E, k_E, lmbda_E_to_price,\
                w, price, price_N = z
            if t == 0:
                lag_i = self.initial["I_N"]/self.initial["K_N"], self.initial["I_E"]/self.initial["K_E"]
                lag_K = self.initial["K_N"], self.initial["K_E"]
            else:
                lag_i = self.i[:,t-1]
                lag_K = self.K[:, t-1]
            system =(
              (1-self.alpha)*price_N  * (self.k[0, t]/l_N_demand)**self.alpha - w/self.A[0, t]\
            )**2+\
            ( 1 - lmbda_N_to_price* (1-self.psi/2 *(i_N/lag_i[0]*self.K[0,t]/lag_K[0] - 1)**2\
               - self.psi * (i_N/lag_i[0]*self.K[0,t]/lag_K[0]) * \
                                      (i_N/lag_i[0]*self.K[0,t]/lag_K[0] - 1) )-\
               self.lmbda_to_price[0,t+1]*self.price[t+1]/price*self.psi/(1+self.r[t+1]) *\
               (self.i[0,t+1]/i_N*k_N*self.A[0, t+1]*self.Labor[t+1]/self.K[0,t] )**2 *\
            (self.i[0,t+1]/i_N*k_N*self.A[0, t+1]*self.Labor[t+1]/self.K[0,t] -1)
            )**2+\
            ( 1/(1+self.r[t+1]) * ((1-self.tau_pi[t+1]) * self.alpha * self.price_N[t+1] *\
                                          (k_N/l_N_demand)**(self.alpha-1) +\
                                          self.tau_pi[t+1] * self.delta * self.price[t+1] +\
                                          self.lmbda_to_price[0, t+1]*self.price[t+1] * (1-self.delta) ) -\
            lmbda_N_to_price * price\
            )**2+\
            ( (1-self.delta)+\
                       i_N * (1-self.psi / 2 *(i_N/lag_i[0]*self.K[0,t]/lag_K[0] - 1)**2 ) -\
            k_N*self.A[0, t+1]*self.Labor[t+1]/self.K[0,t]\
            )**2+\
            (
            (1-self.alpha)*self.price_E[t] * (self.k[1, t]/(1-l_N_demand)*self.A[0,t]/self.A[1,t])**self.alpha - w/ self.A[1, t] 
            )**2+\
            ( 1- lmbda_E_to_price* (1-self.psi/2 *(i_E/lag_i[1]*self.K[1,t]/lag_K[1] - 1)**2\
               - self.psi * (i_E/lag_i[1]*self.K[1,t]/lag_K[1]) * \
                                      (i_E/lag_i[1]*self.K[1,t]/lag_K[1] - 1) )-\
               self.lmbda_to_price[1,t+1]*self.price[t+1]/price*self.psi/(1+self.r[t+1]) *\
               (self.i[1,t+1]/i_E*k_E*self.A[1, t+1]*self.Labor[t+1]/self.K[1,t] )**2 *\
            (self.i[1,t+1]/i_E*k_E*self.A[1, t+1]*self.Labor[t+1]/self.K[1,t] -1)
            )**2+\
            ( 1/(1+self.r[t+1]) * ((1-self.tau_pi[t+1]) * self.alpha * self.price_E[t+1] *\
                                          (k_E/(1-l_N_demand)*self.A[0,t]/self.A[1,t])**(self.alpha-1) +\
                                          self.tau_pi[t+1] * self.delta * self.price[t+1] +\
                                          self.lmbda_to_price[1, t+1]*self.price[t+1] * (1-self.delta) ) -\
            lmbda_E_to_price * price\
            )**2+\
            ((1-self.delta)+\
                       i_E * (1-self.psi / 2 *(i_E/lag_i[1]*self.K[1,t]/lag_K[1] - 1)**2 ) -\
            k_E*self.A[1, t+1]*self.Labor[t+1]/self.K[1,t]
            )**2+\
            ( price - (self.price_M[t])**self.omega * (price_N)**(1-self.omega)\
            )**2+\
            (price_N*((self.k[0,t]/l_N_demand)**self.alpha *l_N_demand\
            - self.Gov[t]/(self.A[0,t]*self.Labor[t])) - \
            (1-self.omega) * price * \
            (self.Consumption[t]/(self.A[0,t]*self.Labor[t]) +\
            i_N*k_N + i_E*k_E+\
            self.psi_O[t] * self.Y_O[t] )
             )**2

            return system



        obj_jit = jit(equilibrium)
        obj_grad = jit(jacfwd(obj_jit))
        obj_hess = jit(jacrev(jacfwd(obj_jit)))

        tol = 1e-5
        result = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=z_guess
                            , options = {"max_iter":self.max_iter, "print_level":0,
                                         "check_derivatives_for_naninf":"yes"}
                            , tol=tol)
        if result["success"] and result['fun']<tol:
            self.i_guess[0,t],self.k_guess[0,t+1],self.l_demand[0,t],\
            self.lmbda_to_price[0,t], self.i_guess[1,t],self.k_guess[1,t+1],\
            self.lmbda_to_price[1,t],\
            self.w[t], self.price_N[t], self.price[t] = self.eta*result["x"] + (1-self.eta)*z_guess
            self.last_guess = z_guess 
        else:
            result
            self.i[0,t],self.k[0,t+1],self.l_demand[0,t],\
            self.lmbda_to_price[0,t], self.i[1,t],self.k[1,t+1],\
            self.lmbda_to_price[1,t],\
            self.w[t], self.price_N[t], self.price[t] = \
            0.5*np.array([self.i[0,t+1],self.k[0,t+2],self.l_demand[0,t+1],\
            self.lmbda_to_price[0,t+1], self.i[1,t+1],self.k[1,t+2],\
            self.lmbda_to_price[1,t+1],\
            self.w[t+1], self.price_N[t+1], self.price[t+1]])+\
            0.5*np.array([self.i[0,t-1],self.k[0,t],self.l_demand[0,t-1],\
            self.lmbda_to_price[0,t-1], self.i[1,t-1],self.k[1,t],\
            self.lmbda_to_price[1,t-1],\
            self.w[t], self.price_N[t-1], self.price[t-1]])
            self.last_guess = None


        self.history[t].append(result)

        self.l_demand[1,t] = 1- self.l_demand[0,t]

        self.L[:,t] = self.l_demand[:,t]*self.Labor[t]

        #self.K[0,t+1] = self.k[0,t+1]*self.Labor[t+1]*self.A[0,t+1]
        #self.K[1,t+1] = self.k[1,t+1]*self.Labor[t+1]*self.A[0,t+1] # тут 0, а не 1!

        #self.I[:,t] = self.i[:,t]*self.K[:,t]

        self.lmbda[:,t] = self.lmbda_to_price[:,t]*self.price[t]
    def  update_capital_guess(self):
        for S in range(2, t_start, t_end):
            self.k[S, t_start:t_end] = self.eta * self.k_guess[S, t_start:t_end]+(1-self.eta)*self.k[S, t_start:t_end]
            self.i[S, t_start:t_end] = self.eta * self.i_guess[S, t_start:t_end]+(1-self.eta)*self.i[S, t_start:t_end]
            self.K[S,1:] = self.k[S, 1:]*self.Labor[1:]*self.A[0,1:]

            self.I[S,:] = self.i[S,:]*self.K[S,:]

