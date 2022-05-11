import pandas as pd
from scipy.optimize import fsolve
import numpy as np
import itertools
import copy
from cyipopt import minimize_ipopt
import warnings
import matplotlib.pyplot as plt
from jax.config import config
# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')

# import jax.numpy as np
from jax import jit, grad, jacfwd, jacrev
from tqdm.contrib.telegram import tqdm, trange
from dotenv import load_dotenv
import os

from bob_telegram_tools.bot import TelegramBot
load_dotenv()

def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped
def labor_income_vector(self, s, g, start, end):
    return (1-self.tau_I[start:end]) * (1-(self.tau_rho[start:end] + self.tau_Ins[start:end])/(1+self.tau_rho[start:end] + self.tau_Ins[start:end])) * self.epsilon[s,g,start:end] * self.w[start:end]


class OLG_model:
    def __init__(self, G,T,N,epsilon, rho, sigma,Pi,r,price_M, price_E, tau_I,tau_II,tau_Ins,
                 tau_pi, tau_VA, tau_rho, beta, phi , theta , psi, omega, alpha,
                 delta, A,initial, eta,steady_max_iter,max_iter,steady_guess):
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
        :param sigma: pension rate (pension point value) as share of wage for T periods
        :param Pi_female: survival probability
        :param Pi_male: survival probability
        :param Beq_female_to_female,Beq_female_to_male,Beq_male_to_female,Beq_male_to_male: bequest matrix
        :param r: rate of return for T periods
        :param price_M: import prices for T periods
        :param price_E: export prices for T periods
        :param tau_I: income tax for T periods
        :param tau_II: income tax on investment for T periods
        :param tau_Ins: insurance tax for T periods
        :param tau_pi: profit tax for T periods
        :param tau_VA: value-added tax for T periods
        :param tau_rho: retirement tax for T periods
        :param beta: elasticity of intertemporal substitution
        :param phi: elasticity of intertemporal substitution
        :param theta: ?
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
        # Retirement
        self.rho, self.sigma = rho, sigma
        # Prices
        self.r, self.price_M, self.price_E, self.omega = r, price_M, price_E, omega
        # Taxation
        self.tau_I, self.tau_II, self.tau_pi, self.tau_VA,self.tau_Ins, self.tau_rho =  tau_I,tau_II, tau_pi,tau_VA,tau_Ins, tau_rho
        # Utility
        self.beta, self.phi, self.theta  = beta, phi, theta
        # Production
        self.psi, self.alpha, self.delta, self.A = psi, alpha, delta, A

        self.a_initial_sum = initial["a_initial_sum"]
        self.a_initial = np.zeros(shape=(2,max_time))
        
        self.initial = initial



        self.eta = eta
        self.steady_max_iter = steady_max_iter
        self.max_iter = max_iter

        # Endogenous variable
        # Initial guess
        self.price_N = np.array([initial["price_N"] for _ in range(max_time)])
        self.K = np.array([[initial["K_N"] for _ in range(max_time)], [initial["K_E"] for _ in range(max_time)]])
        self.L = np.array([[initial["L_N"] for _ in range(max_time)], [initial["L_E"] for _ in range(max_time)]])
        self.I = np.array([[initial["I_N"] for _ in range(max_time)], [initial["I_E"] for _ in range(max_time)]])
        
        self.Debt = np.array([initial["Debt"] for _ in range(max_time)])
        
        
        self.VA_sum= np.zeros_like(self.Debt)
        self.I_sum= np.zeros_like(self.Debt)
        self.II_sum= np.zeros_like(self.Debt)
        
        self.Ins_sum= np.zeros_like(self.Debt)
        self.Rho_sum= np.zeros_like(self.Debt)
        self.Pi_sum= np.zeros_like(self.Debt)
            
        self.Gov_Income= np.zeros_like(self.Debt)
        
        self.Gov_Outcome= np.zeros_like(self.Debt)
        self.Deficit= np.zeros_like(self.Debt)
        self.Deficit_ratio= np.zeros_like(self.Debt)

        self.lmbda = np.array([[0.5 for _ in range(max_time)] for _ in range(2)])

        self.w = ((self.price_N *(1-self.alpha)*(self.K[0])**self.alpha *(self.A[0])**(1-self.alpha)*(self.L[0])**(-self.alpha))+
        (self.price_E *(1-self.alpha)*(self.K[1])**self.alpha *(self.A[1])**(1-self.alpha)*(self.L[1])**(-self.alpha)))/2

        self.price = (self.price_M)**self.omega * (self.price_N)**(1-self.omega)


        self.Y = self.K**self.alpha * (self.L*self.A)**(1-self.alpha)

        
        self.k = np.zeros_like(self.K)
        self.k[0] = self.K[0]/((self.L[0]+self.L[1])*self.A[0])
        self.k[1] = self.K[1]/((self.L[0]+self.L[1])*self.A[1])
        
        self.i = self.I/self.K
        
        self.lmbda_to_price = np.zeros_like(self.lmbda)
        
        self.lmbda_to_price[0] = self.lmbda[0]/self.price
        self.lmbda_to_price[1] = self.lmbda[1]/self.price
        
        self.L_share = np.zeros_like(self.L)
        self.L_share[0] = self.L[0]/(self.L[0]+self.L[1])
        self.L_share[1] = self.L[1]/(self.L[0]+self.L[1])
        


        self.c =  np.array([[[0.5 if ((g >= t) and (g<=self.G+t-1)) else 0 for t in range(max_time)] 
                             for g in range(max_time)] for _ in range(2)])
        self.a = np.array([[[0.1 if ((g >= t) and (g<=self.G+t-1)) else 0 for t in range(max_time)] for g in range(max_time)] for _ in range(2)])
        self.gamma = np.array([[[0.9 if ((g >= t) and (g<=self.G+t-1)) else 0 for t in range(max_time)] for g in range(max_time)] for _ in range(2)])
        self.l = np.array([[[0.4 if ((g >= t) and (g<=self.G+t-1)) else 0 for t in range(max_time)] for g in range(max_time)] for _ in range(2)])

        Consumption = np.array([np.sum([self.c[s,g,t]*self.N[s,g,t] for g in range(max_time) for s in range(2)]) for t in range(max_time)])
        self.Consumption = Consumption

        Labor = np.array([np.sum([self.l[s,g,self.T]*self.N[s,g,self.T]*self.epsilon[s,g,self.T] for g in range(max_time) for s in range(2)]) for t in range(max_time)])
        self.Labor = Labor

        Assets =  np.array([np.sum([self.a[s,g,t]*self.N[s,g,t] for g in range(max_time) for s in range(2)]) for t in range(max_time)])
        self.Assets = Assets

        
        self.steady = np.array(list(steady_guess) + [Consumption[self.T], Labor[self.T], Assets[self.T]])
        self.steady_path = []
        
        self.A_growth = self.A[0,self.T]/self.A[0, self.T-1]
        self.N_growth = np.sum([self.N[s,g,self.T]*self.epsilon[s,g,self.T] for g in range(self.T, self.G+self.T) for s in range(2)])/\
                   np.sum([self.N[s,g,self.T-1]*self.epsilon[s,g,self.T-1] for g in range(self.T-1, self.G+self.T-1) for s in range(2)])
        

        self.lmbda_to_price_steady = (1+self.tau_VA[self.T])/((1 - self.psi/2 * (self.A_growth*self.N_growth -1)**2 - self.psi*self.A_growth*self.N_growth * (self.A_growth*self.N_growth -1))+self.psi/(1+self.r[self.T+1]) * (self.A_growth*self.N_growth -1)* (self.A_growth*self.N_growth)**2)
        self.i_steady  = (self.A_growth*self.N_growth-1+self.delta) / (1-self.psi/2*(self.A_growth*self.N_growth -1)**2)
        
        self.D = self.Consumption+self.I[0]+self.I[1]
        
        self.M = self.omega * self.D * self.price / self.price_M
        
        self.Gov = np.array([self.initial["Gov"]*(self.A_growth*self.N_growth)**i  for i in range(max_time)])

        self.GDP = self.D+ self.price_N/self.price * self.Gov
        
        self.history = {t:[] for t in range(self.T)}
          
    def update_government(self, t):
        self.VA_sum[t] = self.tau_VA[t]*self.price[t]*(self.Consumption[t]+self.I[0,t]+self.I[1, t])
        self.II_sum[t] = self.tau_II[t]*self.Assets[t]
        self.I_sum[t] = self.Labor[t] * self.w[t] *\
                        (1-(self.tau_rho[t] + self.tau_Ins[t])/\
                         (1+self.tau_rho[t] + self.tau_Ins[t])) *\
                        self.tau_I[t] 
        
        
        
        self.Ins_sum[t] = self.tau_Ins[t]/(1+self.tau_rho[t] + self.tau_Ins[t]) * self.Labor[t] * self.w[t]
        self.Rho_sum[t] = self.tau_rho[t]/(1+self.tau_rho[t] + self.tau_Ins[t]) * self.Labor[t] * self.w[t]
        self.Pi_sum[t] = self.tau_pi[t] * (self.price[t] * self.K[0,t]**self.alpha *\
                                           (self.L[0,t]*self.A[0,t])**(1-self.alpha) -\
        self.w[t]*self.L[0,t] - self.delta * self.price[t]*self.K[0,t])+\
            self.tau_pi[t] * (self.price[t] * self.K[1,t]**self.alpha *\
                                           (self.L[1,t]*self.A[1,t])**(1-self.alpha) -\
            self.w[t]*self.L[1,t] - self.delta * self.price[t]*self.K[1,t])
            
        self.Gov_Income[t] = self.VA_sum[t] + self.I_sum[t]+self.II_sum[t]+self.Ins_sum[t]+self.Rho_sum[t]+self.Pi_sum[t]
        if t == 0:
            self.Gov_Outcome[t] = self.Gov[t] + self.sigma[t]*self.w[t] * np.sum(self.rho[:,:,t]*self.N[:,:,t])+\
                self.r[t]*self.initial["Debt"]
        else:
            self.Gov_Outcome[t] = self.Gov[t] + self.sigma[t]*self.w[t] * np.sum(self.rho[:,:,t]*self.N[:,:,t])+\
                self.r[t]*self.Debt[t-1]
        self.Deficit[t] = self.Gov_Outcome[t] - self.Gov_Income[t]
        self.Deficit_ratio[t] = self.Deficit[t]/(self.Consumption[t]+np.sum(self.I[:,t])+self.Gov[t])
        if t == 0:
            self.Debt[t] = self.initial["Debt"]+self.Deficit[t]
        else:
            self.Debt[t] = self.Debt[t-1]+self.Deficit[t]
        
        


    def household(self, s, g, t, t_0=None):
        def cumulative_rate_of_return(self,start, end):
            if start <= end:
                return np.prod(1+self.r[start:end]*(1-self.tau_II[start:end]))
            else:
                return 1

        def get_initial_consumption(self, s, g, t_0):
            return self.phi * (self.a_initial[s,g] * cumulative_rate_of_return(self, t_0, g+1)+
                               np.sum(np.array([cumulative_rate_of_return(self,start, g+1) for start in range(t_0+1, g+2)])*
                                      (labor_income_vector(self, s, g, t_0, g+1)+self.rho[s,g,t_0:(g+1)]*self.sigma[t_0:(g+1)]*self.w[t_0:(g+1)]
                                       )))/\
                   np.sum(
                       np.array([cumulative_rate_of_return(self,start, g+1) for start in range(t_0+1, g+2)])*
                       (1+self.tau_VA[t_0:(g+1)])*(self.price[t_0:(g+1)])*
                       (
                               np.array([cumulative_rate_of_return(self,t_0+1, end) for end in range(t_0+1, g+2)])*
                               self.beta**np.array([i-t_0 for i in range(t_0, g+1)])*self.Pi[s,g,t_0:(g+1)]/self.Pi[s,g,t_0] * (1+self.tau_VA[t_0])*(self.price[t_0])/((1+self.tau_VA[t_0:(g+1)])*(self.price[t_0:(g+1)])) *
                               ((1+self.tau_VA[t_0:(g+1)])*(self.price[t_0:(g+1)])/((1+self.tau_VA[t_0])*self.price[t_0])*labor_income_vector(self, s, g,t_0, t_0+1)/labor_income_vector(self, s, g,t_0, g+1)
                                )**((1-self.phi)*(1-self.theta)))**(1/self.theta)
                   )
        if t_0 is None:
            t_0 = max(g-self.G+1,0)
        bequest = 0
        if t==t_0:
            consumption = get_initial_consumption(self, s, g, t_0)
        elif t_0 < t <= g:
            consumption = self.c[s, g, t_0]*\
                          (cumulative_rate_of_return(self, t_0+1, t+1)*
                           self.beta**(t-t_0)*self.Pi[s,g,t]/self.Pi[s,g,t_0] * (1+self.tau_VA[t_0])*\
                           self.price[t_0]/((1+self.tau_VA[t])*self.price[t]) *
                           ((1+self.tau_VA[t])*self.price[t]/((1+self.tau_VA[t_0])*self.price[t_0])*\
                            labor_income_vector(self, s, g,t_0, t_0+1)/labor_income_vector(self, s, g,t, t+1))**((1-self.phi)*(1-self.theta))
                           )**(1/self.theta)


        labor = 1- consumption/(self.phi/(1-self.phi)*(1/((1+self.tau_VA[t])*self.price[t]))*labor_income_vector(self, s, g, t, t+1)[0])
        if t == 0:
            assets = labor_income_vector(self, s, g, t, t+1)[0]*labor+self.rho[s,g,t]*self.sigma[t]*self.w[t] - consumption*(1+self.tau_VA[t])*self.price[t]+self.a_initial[s,g]*(1+self.r[t]*(1-self.tau_II[t]))
        else:
            assets = labor_income_vector(self, s, g, t, t+1)[0]*labor+self.rho[s,g,t]*self.sigma[t]*self.w[t] - consumption*(1+self.tau_VA[t])*self.price[t]+(self.a[s,g, t-1]+bequest)*(1+self.r[t]*(1-self.tau_I[t]))
        return consumption, labor, assets

    @counted
    def steady_state(self):

        w_steady, price_steady, price_N_steady = self.steady[3:6]

        for t in range(self.T-self.G + 1, max_time):
            self.w[t] = w_steady* self.A_growth**(t - self.T)
            self.price[t] = price_steady
            self.price_N[t] = price_N_steady

        for s in range(2):
            initial_household = self.household(s, self.G+self.T-1, self.T) # первое родившееся поколение в период self.T
            self.c[s, self.G+self.T-1, self.T],self.l[s, self.G+self.T-1,self.T],self.a[s, self.G+self.T-1,self.T] = initial_household
            for t in range(self.T,self.G+self.T):
                self.c[s, self.G+self.T-1, t],self.l[s, self.G+self.T-1,t],self.a[s, self.G+self.T-1,t] = self.household(s, self.G+self.T-1, t)
            for g in range(self.G+self.T-2, self.T-1, -1): # все поколения, живущие в периоде self.T

                self.c[s, g, g-self.G + 1]  = initial_household[0] * self.A_growth**(self.G+self.T-1 - g)
                self.l[s, g, g-self.G + 1]  = initial_household[1]
                self.a[s, g,g-self.G + 1]  = labor_income_vector(self, s, g,g-self.G + 1, g-self.G + 2)[0]*self.l[s, g, g-self.G + 1]+self.rho[s,g,g-self.G + 1]*self.w[g-self.G + 1]*self.sigma[g-self.G + 1] - self.c[s, g, g-self.G + 1]*(1+self.tau_VA[g-self.G + 1])*self.price[g-self.G + 1]
                for t in range(g-self.G + 2, g+1):
                    self.c[s, g, t],self.l[s, g, t],self.a[s, g, t] =  self.household(s, g, t)
        Consumption = np.sum([self.c[s,g,self.T]*self.N[s,g,self.T] 
                              for g in range(self.T, self.G+self.T) 
                              for s in range(2)])

        Labor = np.sum([self.l[s,g,self.T]*self.N[s,g,self.T]*self.epsilon[s,g,self.T] for g in range(self.T, self.G+self.T) for s in range(2)])

        Assets =  np.sum([self.a[s,g,self.T]*self.N[s,g,self.T] for g in range(self.T, self.G+self.T) for s in range(2)])

        if len(self.steady_path)==0:
            self.steady[-3:] = np.array([Consumption,  Labor, Assets])
        else:
            self.steady[-3:] = self.eta*np.array([Consumption,  Labor, Assets]) + (1-self.eta)*self.steady[-3:]

        z_guess = self.steady[:6]
        
        def equilibrium(z, self=self, objective = True):


  
            system = [
                f"{1-self.alpha}*price_N_steady * (k_N_steady/L_N_share)**{self.alpha} *{self.A[0,self.T]} - w_steady"
                      ,f"{1/(1+self.r[self.T+1])}*(({1-self.tau_pi[self.T+1]}) *{self.alpha}*price_N_steady* (k_N_steady/L_N_share)**{self.alpha-1} +{self.tau_pi[self.T+1]} * {self.delta}*price_steady + {self.lmbda_to_price_steady}*price_steady * ({1-self.delta})) - {self.lmbda_to_price_steady}*price_steady"
                      ,f"{1-self.alpha}*{self.price_E[self.T]} * (k_E_steady/(1-L_N_share)*{self.A[0,self.T]/self.A[1,self.T]} )**{self.alpha} *{self.A[1,self.T]} - w_steady"
                      ,f"{1/(1+self.r[self.T+1])}*({1-self.tau_pi[self.T+1]}*{self.alpha}* {self.price_E[self.T+1]}* (k_E_steady/(1-L_N_share)*{self.A[0,self.T]/self.A[1,self.T]})**{self.alpha-1} +{self.tau_pi[self.T+1]} * {self.delta}*price_steady +{self.lmbda_to_price_steady}*price_steady * {1-self.delta}) - {self.lmbda_to_price_steady}*price_steady"
                      ,f"price_N_steady*((k_N_steady/L_N_share)**{self.alpha} * L_N_share -{self.Gov[self.T]/(self.A[0,self.T]*self.steady[-2])})- (1-{self.omega}) * price_steady * ({self.steady[-3]/(self.A[0,self.T]*self.steady[-2])}+{self.i_steady}*(k_N_steady+k_E_steady))"
                      ,f"price_steady - {self.price_M[self.T]**self.omega}*price_N_steady**({1-self.omega})"
                      ]
            name_space = {label:value for label, value in zip(("self", "k_N_steady", "L_N_share",
                                                               "k_E_steady", "w_steady",
                                                               "price_steady", "price_N_steady"),[self]+list(z))}

            if objective:
                sum_of_squares = "+".join([f"({equation})**2" for equation in system])
                F = eval(sum_of_squares,{},name_space)
            else:
                F = [eval(equation,{},name_space) for equation in system]
            return F
        def equilibrium_gradient(z, self=self, objective = True):

            self.A_growth = self.A[0,self.T]/self.A[0, self.T-1]
            self.N_growth = np.sum([self.N[s,g,self.T]*self.epsilon[s,g,self.T] for g in range(self.T, self.G+self.T) for s in range(2)])/\
                       np.sum([self.N[s,g,self.T-1]*self.epsilon[s,g,self.T-1] for g in range(self.T-1, self.G+self.T-1) for s in range(2)])

            name_space = {label:value for label, value in zip(("self","I_N_steady", "K_N_steady", "L_N_steady","I_E_steady", "K_E_steady", "lmbda_steady", "w_steady", "price_steady", "price_N_steady"),[self]+list(z))}
            if objective:
                target_system = equilibrium(z, self, False)
                sum_of_gradient = ["+".join([f"2*({outer})*({inner})" for outer, inner in zip(target_system, row)]) for row in gradient]
                J = [eval(equation,{},name_space) for equation in sum_of_gradient]
            else:
                J = [[eval(equation,{},name_space) for equation in row] for row in gradient]
            return J
            

        
        obj_jit = jit(equilibrium)
        obj_grad = jit(jacfwd(obj_jit))
        obj_hess = jit(jacrev(jacfwd(obj_jit)))
        
        
        result = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=z_guess
                                , options = {"max_iter":self.steady_max_iter, "print_level":0,
                                             "check_derivatives_for_naninf":"yes"}
                                , tol=1e-7)
        
        
        if result["success"] or result["status"]==1:
            self.steady[:6] = self.eta*result["x"] + (1-self.eta)*z_guess
            self.steady_max_iter = 1000
            eq_res = equilibrium(self.steady[:6], self, False)
        else:
            if result["status"]==-1:
                self.steady_max_iter += 1000
            self.steady[:6] = z_guess
            eq_res = equilibrium(z_guess, self, False)
            
                
        self.steady_path.append((result, np.array(self.steady)
                               ))

    def create_guess(self):
        
        k_N_steady, L_N_share, k_E_steady, w_steady, price_steady, price_N_steady, Consumption_steady, Labor_steady, Assets_steady = self.steady
        
        steady_start = self.T#-self.G + 1
        
        for t in range(steady_start, max_time):
            self.update_government(t)
            
            self.w[t] = w_steady* self.A_growth**(t - self.T)
            self.price[t] = price_steady
            self.price_N[t] = price_N_steady
            
            self.k[0,t] = k_N_steady
            self.k[1,t] = k_E_steady
            
            self.i[0,t] = self.i_steady
            self.i[1,t] = self.i_steady 
            
            self.L_share[0,t] = L_N_share 
            self.L_share[1,t] = 1- L_N_share 
            
            self.lmbda_to_price[0,t] = self.lmbda_to_price_steady
            self.lmbda_to_price[1,t] = self.lmbda_to_price_steady
            
            
            
            self.K[0,t] = k_N_steady* Labor_steady * self.A[0,self.T] *\
            (self.A_growth*self.N_growth)**(t - self.T)
            self.K[1,t] = k_E_steady* Labor_steady * self.A[0,self.T] *\
            (self.A_growth*self.N_growth)**(t - self.T)
            
            self.I[0,t] = self.i_steady * self.K[0,t]
            self.I[1,t] = self.i_steady * self.K[1,t]
            
            self.L[0,t] = L_N_share * Labor_steady *(self.N_growth)**(t - self.T)
            self.L[1,t] = (1-L_N_share) * Labor_steady *(self.N_growth)**(t - self.T)
            
            self.lmbda[0,t] = self.lmbda_to_price_steady * price_steady
            self.lmbda[1,t] = self.lmbda_to_price_steady * price_steady
            
            self.Consumption[t] = Consumption_steady*(self.A_growth*self.N_growth)**(t - self.T)
            self.Labor[t] = Labor_steady*(self.N_growth)**(t - self.T)
            self.Assets[t] =  Assets_steady*(self.A_growth*self.N_growth)**(t - self.T)
        
        
        k_N_initial = self.initial["K_N"]/((self.initial["L_N"]+self.initial["L_E"])*self.A[0,0])
        k_E_initial = self.initial["K_E"]/((self.initial["L_N"]+self.initial["L_E"])*self.A[0,0])
        
        self.k[0,:steady_start] = np.linspace(k_N_initial, k_N_steady,\
                                                  steady_start,endpoint=False)
        self.k[1,:steady_start] = np.linspace(k_E_initial, k_E_steady,\
                                                  steady_start,endpoint=False)
        
        i_N_initial = self.initial["I_N"]/self.initial["K_N"]
        i_E_initial = self.initial["I_E"]/self.initial["K_E"]
        self.i[0,:steady_start] = np.linspace(i_N_initial, self.i_steady,\
                                                  steady_start,endpoint=False)
        self.i[1,:steady_start] = np.linspace(i_E_initial, self.i_steady,\
                                                  steady_start,endpoint=False)
        L_share_N_initial = self.initial["L_N"]/(self.initial["L_N"]+self.initial["L_E"])
        L_share_E_initial = self.initial["L_E"]/(self.initial["L_N"]+self.initial["L_E"])
        
        self.L_share[0,:steady_start] = np.linspace(L_share_N_initial, L_N_share,\
                                                  steady_start,endpoint=False)
        self.L_share[1,:steady_start] = np.linspace(L_share_E_initial, 1-L_N_share,\
                                                  steady_start,endpoint=False)
            
        self.lmbda_to_price[0,:steady_start] = self.lmbda_to_price_steady
        self.lmbda_to_price[1,:steady_start] = self.lmbda_to_price_steady
        
        self.price_N[:steady_start] = np.linspace(self.initial["price_N"], self.price_N[steady_start],\
                                                  steady_start,endpoint=False)
        w_initial = self.price_N[0] *(1-self.alpha)*(self.k[0,0])**self.alpha *(self.A[0,0])**(1-self.alpha)
        
        self.w[:steady_start] = np.linspace(w_initial, w_steady, steady_start,endpoint=False)
        self.price[:steady_start] = (self.price_M[:steady_start])**self.omega *\
        (self.price_N[:steady_start])**(1-self.omega)

        
        self.I[0,:steady_start] = np.linspace(self.initial["I_N"], self.I[0,steady_start], steady_start,endpoint=False)
        self.K[0,:steady_start] = np.linspace(self.initial["K_N"], self.K[0,steady_start], steady_start,endpoint=False)
        self.L[0,:steady_start] = np.linspace(self.initial["L_N"], self.L[0,steady_start], steady_start,endpoint=False)

        self.I[1,:steady_start] = np.linspace(self.initial["I_E"], self.I[1,steady_start], steady_start,endpoint=False)
        self.K[1,:steady_start] = np.linspace(self.initial["K_E"], self.K[1,steady_start], steady_start,endpoint=False)
        self.L[1,:steady_start] = np.linspace(self.initial["L_E"], self.L[1,steady_start], steady_start,endpoint=False)

        self.lmbda[0,:steady_start] = self.price[:steady_start]*self.lmbda_to_price_steady
        self.lmbda[1,:steady_start] = self.price[:steady_start]*self.lmbda_to_price_steady
        
        
        for t in range(self.T):
            for s in range(2):
                for g in range(t, self.G+t):
                    self.c[s, g, t], self.l[s,g,t], self.a[s,g,t] = self.household(s,g,t)
                    
        self.Consumption[:self.T] = np.array([np.sum(self.c[:,:,t]*self.N[:,:self.c.shape[1],t]) for t in range(self.T)])
        self.Labor[:self.T] = np.array([np.sum(self.l[:,:,t]*self.N[:,:self.l.shape[1],t]*self.epsilon[:,:self.l.shape[1],t]) for t in range(self.T)])
        self.Assets[:self.T] = np.array([np.sum(self.a[:,:,t]*self.N[:,:self.a.shape[1],t]) for t in range(self.T)])
        
        

        

        

    def update_a_initial(self):
        coef = np.sum(self.a[:,self.G+self.T-2:self.T-1:-1,self.T]*\
                      self.N[:,self.G+0-2::-1,0])/self.a_initial_sum
        
        self.a_initial[:,self.G-2::-1]=self.a[:,self.G+self.T-2:self.T-1:-1,self.T]/coef
        

    def update_guess(self, t):
        
        self.update_government(t)
        for s in range(2):
            for g in range(t, self.G+t):
                self.c[s, g, t],self.l[s, g, t],self.a[s, g,t]  = self.household(s,g,t)
                
        self.Consumption[t] = np.sum([self.c[s,g,t]*self.N[s,g,t] 
                                      for g in range(t, self.G+t) 
                                      for s in range(2)])
        self.Labor[t] = np.sum([self.l[s,g,t]*self.N[s,g,t]*self.epsilon[s,g,t] 
                                for g in range(t, self.G+t) 
                                for s in range(2)])
        self.Assets[t] = np.sum([self.a[s,g,t]*self.N[s,g,t]
                                 for g in range(t, self.G+t) 
                                 for s in range(2)])
        
        z_guess = np.array([self.i[0,t],self.k[0,t+1],self.L_share[0,t],
                   self.lmbda_to_price[0,t], self.i[1,t],self.k[1,t+1],
                   self.lmbda_to_price[1,t],
                   self.w[t], self.price_N[t], self.price[t]])


        def equilibrium(z, self=self, t = t):
            i_N, k_N, L_N_share, lmbda_N_to_price,\
                i_E, k_E, lmbda_E_to_price,\
                w, price, price_N = z
            if t == 0:
                lag_i = self.initial["I_N"]/self.initial["K_N"], self.initial["I_E"]/self.initial["K_E"]
                lag_K = self.initial["K_N"], self.initial["K_E"]
            else:
                lag_i = self.i[:,t-1]
                lag_K = self.K[:, t-1]
            system =(
              (1-self.alpha)*price_N * self.A[0, t] * (self.k[0, t]/L_N_share)**self.alpha - w\
            )**2+\
            ( (1+tau_VA[t])*price - lmbda_N_to_price * price* (1-self.psi/2 *(i_N/lag_i[0]*self.K[0,t]/lag_K[0] - 1)**2\
               - self.psi * (i_N/lag_i[0]*self.K[0,t]/lag_K[0]) * \
                                      (i_N/lag_i[0]*self.K[0,t]/lag_K[0] - 1) )-\
               self.lmbda_to_price[0,t+1]*self.price[t+1]*self.psi/(1+self.r[t+1]) *\
               (self.i[0,t+1]/i_N*k_N*self.A[0, t+1]*self.Labor[t+1]/self.K[0,t] )**2 *\
            (self.i[0,t+1]/i_N*k_N*self.A[0, t+1]*self.Labor[t+1]/self.K[0,t] -1)
            )**2+\
            ( 1/(1+self.r[t+1]) * ((1-self.tau_pi[t+1]) * self.alpha * self.price_N[t+1] *\
                                          (k_N/L_N_share)**(self.alpha-1) +\
                                          self.tau_pi[t+1] * self.delta * self.price[t+1] +\
                                          self.lmbda_to_price[0, t+1]*self.price[t+1] * (1-self.delta) ) -\
            lmbda_N_to_price * price\
            )**2+\
            ( (1-self.delta)+\
                       i_N * (1-self.psi / 2 *(i_N/lag_i[0]*self.K[0,t]/lag_K[0] - 1)**2 ) -\
            k_N*self.A[0, t+1]*self.Labor[t+1]/self.K[0,t]\
            )**2+\
            (
            (1-self.alpha)*self.price_E[t] * self.A[1, t] * (self.k[1, t]/(1-L_N_share)*self.A[0,t]/self.A[1,t])**self.alpha - w
            )**2+\
            ( (1+tau_VA[t])*price - lmbda_E_to_price*price * (1-self.psi/2 *(i_E/lag_i[1]*self.K[1,t]/lag_K[1] - 1)**2\
               - self.psi * (i_E/lag_i[1]*self.K[1,t]/lag_K[1]) * \
                                      (i_E/lag_i[1]*self.K[1,t]/lag_K[1] - 1) )-\
               self.lmbda_to_price[1,t+1]*self.price[t+1]*self.psi/(1+self.r[t+1]) *\
               (self.i[1,t+1]/i_E*k_E*self.A[1, t+1]*self.Labor[t+1]/self.K[1,t] )**2 *\
            (self.i[1,t+1]/i_E*k_E*self.A[1, t+1]*self.Labor[t+1]/self.K[1,t] -1)
            )**2+\
            ( 1/(1+self.r[t+1]) * ((1-self.tau_pi[t+1]) * self.alpha * self.price_E[t+1] *\
                                          (k_E/(1-L_N_share)*self.A[0,t]/self.A[1,t])**(self.alpha-1) +\
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
            (price_N*((self.k[0,t]/L_N_share)**self.alpha *L_N_share\
            - self.Gov[t]/(self.A[0,t]*self.Labor[t])) - \
            (1-self.omega) * price * \
            (self.Consumption[t]/(self.A[0,t]*self.Labor[t]) +\
            i_N*k_N + i_E*k_E)
             )**2

            return system



        obj_jit = jit(equilibrium)
        obj_grad = jit(jacfwd(obj_jit))
        obj_hess = jit(jacrev(jacfwd(obj_jit)))


        result = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=z_guess
                            , options = {"max_iter":self.max_iter, "print_level":0,
                                         "check_derivatives_for_naninf":"yes"}
                            , tol=1e-4)
        if result["success"]:
            self.i[0,t],self.k[0,t+1],self.L_share[0,t],\
            self.lmbda_to_price[0,t], self.i[1,t],self.k[1,t+1],\
            self.lmbda_to_price[1,t],\
            self.w[t], self.price_N[t], self.price[t] = self.eta*result["x"] + (1-self.eta)*z_guess
        else:
            self.i[0,t],self.k[0,t+1],self.L_share[0,t],\
            self.lmbda_to_price[0,t], self.i[1,t],self.k[1,t+1],\
            self.lmbda_to_price[1,t],\
            self.w[t], self.price_N[t], self.price[t] = \
            0.5*np.array([self.i[0,t+1],self.k[0,t+2],self.L_share[0,t+1],\
            self.lmbda_to_price[0,t+1], self.i[1,t+1],self.k[1,t+2],\
            self.lmbda_to_price[1,t+1],\
            self.w[t+1], self.price_N[t+1], self.price[t+1]])+\
            0.5*np.array([self.i[0,t],self.k[0,t+1],self.L_share[0,t],\
            self.lmbda_to_price[0,t], self.i[1,t],self.k[1,t+1],\
            self.lmbda_to_price[1,t],\
            self.w[t], self.price_N[t], self.price[t]])
            
        
        self.history[t].append(result)
        
        self.L_share[1,t] = 1- self.L_share[0,t]

        self.L[:,t] = self.L_share[:,t]*self.Labor[t]

        self.K[0,t+1] = self.k[0,t+1]*self.Labor[t+1]*self.A[0,t+1]
        self.K[1,t+1] = self.k[1,t+1]*self.Labor[t+1]*self.A[0,t+1] # тут 0, а не 1!

        self.I[:,t] = self.i[:,t]*self.K[:,t]

        self.lmbda[:,t] = self.lmbda_to_price[:,t]*self.price[t]

        self.Y[:,t] = self.K[:,t]**self.alpha * (self.L[:,t]*self.A[:,t])**(1-self.alpha)
        self.D[t] = self.Consumption[t]+self.I[0,t]+self.I[1,t]
        self.M[t] =  self.omega * self.D[t] * self.price[t] / self.price_M[t]

        
           
demography = np.load('demography.npy',allow_pickle='TRUE').item()
n_generations = demography["N_female"][0][demography["N_female"][0]!=0].shape[0]


epsilon_female =epsilon_male= np.array([[np.exp(-0.001*(n_generations-g+t)**2+15*0.001*2*(n_generations-g+t))\
                                         if ((n_generations-g+t) > 0 and (n_generations-g+t)<=n_generations) \
                                         else 0  for g in range(demography["N_female"].shape[1]) ] \
                                        for t in range(demography["N_female"].shape[0])])


rho_female = np.array([[1 if ((n_generations-g+t) >= 46 and (n_generations-g+t)<=n_generations) else 0  for g in range(demography["N_female"].shape[1]) ] for t in range(demography["N_female"].shape[0])])
rho_male = np.array([[1 if ((n_generations-g+t) >= 51 and (n_generations-g+t)<=n_generations) else 0  for g in range(demography["N_female"].shape[1]) ] for t in range(demography["N_female"].shape[0])])

max_time = demography["N_female"].shape[0]
sigma = np.array([0.9 for _ in range(max_time)])
r = np.array([0.03 for _ in range(max_time)])
price_M = np.array([1. for _ in range(max_time)])
price_E = np.array([1. for _ in range(max_time)])
tau_I = np.array([0.13 for _ in range(max_time)])
tau_II = np.array([0. for _ in range(max_time)])
tau_Ins = np.array([0.08 for _ in range(max_time)])
tau_pi = np.array([0.2 for _ in range(max_time)])
tau_VA = np.array([0.18 for _ in range(max_time)])
tau_rho = np.array([0.22 for _ in range(max_time)])
A_N = np.array([1. for _ in range(max_time)])
A_E = np.array([1. for _ in range(max_time)])


N = np.array([demography["N_female"].transpose()/1000,demography["N_male"].transpose()/1000])
Pi = np.array([(demography[f"N_{sex}male"]/np.max(demography[f"N_{sex}male"], axis=0)\
                +(1-np.tri(*demography[f"N_{sex}male"].shape, k=n_generations-1))).transpose() for sex in ("fe", "")])
Pi[Pi<0] = 0.01
epsilon = np.array([epsilon_female.transpose(), epsilon_male.transpose()])
rho = np.array([rho_female.transpose(), rho_male.transpose()])
A = np.array([A_N, A_E])
initial = {"a_initial_sum":-13,
                "price_N":1.,
                 "K_N":400.,
                 "L_N":35.,
                 "I_N":3,
                 "K_E":600.,
                 "L_E":50.,
                 "Gov":7., # растет с темпом A_growth * N_growth
                 "Debt":-1.,
                 "I_E":5}
steady_guess = np.array([5.54349368,    0.38667374,    8.7962513 ,    1.61917223,
           0.99999988,    0.9999998])
# steady state
# [ 1.55169913e+00,  5.05142101e-02,  2.91663738e+01,  2.08788575e-01,
#         1.00000000e-01,  1.00000001e-01,  4.82953469e-02,  8.82916686e+01,
#        -6.42938235e+01]


            
