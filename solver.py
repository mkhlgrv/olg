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
import pyreadr
import numpy as np
import pickle
import pandas as pd
import statsmodels.api as sm
ranepa_colors = [(146/250,26/250,29/250)
,(230/250 , 43/250, 37/250)
,(231/250, 142/250, 36/250)
,(249/250, 155/250, 28/250)
, (242/250, 103/250, 36/250)]

from olg import *

steady_guess = np.array([2.65280161e+00, 7.20838767e-01, 1.02735785e+00, 3.30768049e+01,
       1.00000000e+00, 1.00000000e+00, 4.67438156e-01, 3.66973833e+05,
       1.64488185e+04, 2.78606328e+05])

import jax.numpy as jnp
import sys

t_0=1
t_1 = int(sys.argv[1])

olg = OLG_model(T = 300,eta=0.5,utility = "exogenous_labor",steady_guess = steady_guess)

olg.update_a_initial()
olg.create_guess(t_0=t_0, steady_start=t_1)

class JAX_OLG():
    def copy_from(self, model):
        self.price_M = jnp.array(model.price_M).copy()
        self.A = jnp.array(model.A).copy()
        self.Labor = jnp.array(model.Labor).copy()
        self.r = jnp.array(model.r).copy()
        self.tau_pi = jnp.array(model.tau_pi).copy()
        self.Gov = jnp.array(model.Gov).copy()
        self.Consumption = jnp.array(model.Consumption).copy()
        self.psi_O =jnp.array(model.psi_O).copy()
        self.Y_O = jnp.array(model.Y_O).copy()
        
# static
static = JAX_OLG()
static.copy_from(olg)



def equilibrium(z,  t_0=1,t_1 = t_1, self=olg):
    period = t_1 - t_0

    # border conditions ( from t = t_0 to 

    k = jnp.array(self.k).copy()
    k = k.at[:, (t_0+1):(t_1+1)].set(z[:(period*2)].reshape(2, -1))

    i = jnp.array(self.i).copy()
    i = i.at[:, t_0:t_1].set(z[(period*2):(period*4)].reshape(2, -1))

    l_demand = jnp.array(self.l_demand).copy()
    l_demand = l_demand.at[0, t_0:t_1].set(z[(period*4):(period*5)])
    l_demand = l_demand.at[1, t_0:t_1].set(1 - z[(period*4):(period*5)])

    lmbda_to_price = jnp.array(self.lmbda_to_price).copy()
    lmbda_to_price = lmbda_to_price.at[:, t_0:t_1].set(z[(period*5):(period*7)].reshape(2, -1))

    w = jnp.array(self.w).copy()
    w = w.at[t_0:t_1].set(z[(period*7):(period*8)])

    price = jnp.array(self.price).copy()
    price = price.at[t_0:t_1].set(z[(period*8):(period*9)])

    price_S = jnp.array([self.price_N.copy(), self.price_E.copy()])
    price_S = price_S.at[0, t_0:t_1].set(z[(period*9):(period*10)])
    

    
    
    


    def labor_equation(t, S, self=self):
        return (1-self.alpha)*price_S[S, t]  * (k[S, t]/l_demand[S, t] * static.A[0,t]/static.A[S, t])**self.alpha - w[t]/static.A[0, t]

    def investment_equation(t, S, self=self):
        return  1 -\
                lmbda_to_price[S, t] * (
                                    1 - self.psi/2 *(
                                                    i[S, t]/i[S, t-1]*(k[S,t] * static.Labor[t] * static.A[0, t])/\
                                                                      (k[S,t-1] * static.Labor[t-1] * static.A[0, t-1])
                                                    - 1
                                                    )**2\
                                      - self.psi * (
                                                    i[S, t]/i[S, t-1]*(k[S,t] * static.Labor[t] * static.A[0, t])/\
                                                                      (k[S,t-1] * static.Labor[t-1] * static.A[0, t-1])
                                                    ) * \
                                                   (
                                                    i[S, t]/i[S, t-1]*(k[S,t] * static.Labor[t] * static.A[0, t])/\
                                                                      (k[S,t-1] * static.Labor[t-1] * static.A[0, t-1]) - 1
                                                   ) 
                                    )-\
               lmbda_to_price[S,t+1]*price[t+1]/price[t]*self.psi/(1+static.r[t+1]) *\
               (
                    i[S,t+1]/i[S,t]*(k[S, t+1]*static.A[0, t+1]*static.Labor[t+1])/\
                                    (k[S,t] * static.Labor[t] * static.A[0, t])
               )**2 *\
               (
                    i[S,t+1]/i[S,t]*(k[S, t+1]*static.A[0, t+1]*static.Labor[t+1])/\
                                    (k[S,t] * static.Labor[t] * static.A[0, t])
                    - 1
               )

    def capital_equation(t, S, self=self):
        return 1/(1+static.r[t+1]) * (
                                      (1-static.tau_pi[t+1]) * self.alpha * price_S[S, t+1] *\
                                      (k[S, t]/l_demand[S, t] * static.A[0,t]/static.A[S, t])**(self.alpha-1) +\
                                      static.tau_pi[t+1] * self.delta * price[t+1] +\
                                      lmbda_to_price[S, t+1]*price[t+1] * (1-self.delta) 
                                    ) -\
                lmbda_to_price[S, t] * price[t]
    def capital_accomodation_equation(t, S, self=self):
        return (1-self.delta)+\
                i[S,t] * (1-self.psi / 2 *(i[S, t]/i[S, t-1]*(k[S,t] * static.Labor[t] * static.A[0, t])/\
                                                             (k[S,t-1] * static.Labor[t-1] * static.A[0, t-1])
                                           - 1
                                          )**2 
                         ) -\
                k[S,t+1]*static.A[0, t+1]*static.Labor[t+1]/(k[S,t] * static.Labor[t] * static.A[0, t])


    
    def final_good_market_equation(t, self=self):
        return price[t] - (static.price_M[t])**self.omega * (price_S[0, t])**(1-self.omega)

    def intermediate_good_market_equation(t, self=self):
        return price_S[0,t]*(
                                (k[0,t]/l_demand[0,t])**self.alpha *l_demand[0,t] - \
                                static.Gov[t]/(static.A[0,t]*static.Labor[t])
                            ) - \
                (1-self.omega) * price[t] * \
                (
                    (static.Consumption[t] + static.psi_O[t] * static.Y_O[t])/(static.A[0,t]*static.Labor[t]) +\
                    i[0,t]*k[0,t] + i[1,t]*k[1,t]
                )

    
    def system(t):
        return  ( 
                labor_equation(t,  0)**2+
                labor_equation(t,  1)**2+
                investment_equation(t,  0)**2+
                investment_equation(t,  1)**2+
                capital_equation(t,  0)**2 +
                capital_equation(t,  1)**2 +
                capital_accomodation_equation(t,  0)**2+
                capital_accomodation_equation(t,  1)**2+
                final_good_market_equation( t)**2+
                intermediate_good_market_equation( t)**2
        )
    
    
    return jnp.apply_along_axis(system, 0, jnp.arange(t_0, t_1)).sum()


def update_guess(self, t_0, t_1):
    z = np.concatenate([self.k[:, (t_0+1):(t_1+1)].ravel(),
                  self.i[:, t_0:t_1].ravel(),
                  self.l_demand[0, t_0:t_1].ravel(),
                  self.lmbda_to_price[:, t_0:t_1].ravel(),
                  self.w[t_0:t_1],
                  self.price[t_0:t_1],
                  self.price_N[t_0:t_1]
                 ])
    return z


def update_optimal_trajectory(self, z, t_0, t_1):
    self.k[0, (t_0+1):(t_1+1)],\
    self.k[1, (t_0+1):(t_1+1)],\
    self.i[0, t_0:t_1],\
    self.i[1, t_0:t_1],\
    self.l_demand[0, t_0:t_1],\
    self.lmbda_to_price[0, t_0:t_1],\
    self.lmbda_to_price[1, t_0:t_1],\
    self.w[t_0:t_1],\
    self.price[t_0:t_1],\
    self.price_N[t_0:t_1] = np.split(z, 10)
    return self


z_guess = update_guess(olg, t_0, t_1)


import cyipopt
import sparsejac
import jax

get_diagonal = lambda k: jax.numpy.eye(N = int(t_1-t_0), k=k, dtype=int)
hessian_diagonal = get_diagonal(-1)+get_diagonal(0)+get_diagonal(1)
hessian_structure = jnp.nonzero(jnp.tile(hessian_diagonal, reps = (int(len(z_guess)/(t_1-t_0)),int(len(z_guess)/(t_1-t_0)))))

#obj_jit = jit(equilibrium)
#obj_grad = jit(jacfwd(obj_jit))
#obj_hess= jit(jacrev(jacfwd(obj_jit)))
obj_jit = equilibrium
obj_grad = jacfwd(obj_jit)
obj_hess = jacrev(jacfwd(obj_jit))


class OLG_problem():

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return obj_jit(x)

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return obj_grad(x)

#     def constraints(self, x):
#         """Returns the constraints."""
#         return np.array((np.prod(x), np.dot(x, x)))

#     def jacobian(self, x):
#         """Returns the Jacobian of the constraints with respect to x."""
#         return np.concatenate((np.prod(x)/x, 2*x))

    def hessianstructure(self):
        """Returns the row and column indices for non-zero vales of the
        Hessian."""

        return hessian_structure

    def hessian(self, x, lagrange, obj_factor):
        """Returns the non-zero values of the Hessian."""
        row, col = self.hessianstructure()
        
        return obj_hess(x)[row, col]

#     def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
#                      d_norm, regularization_size, alpha_du, alpha_pr,
#                      ls_trials):
#         """Prints information at every Ipopt iteration."""

#         msg = "Objective value at iteration #{:d} is - {:g}"

#         print(msg.format(iter_count, obj_value))
nlp = cyipopt.Problem(
   n=len(z_guess),
   m=0,
   problem_obj=OLG_problem(),
   lb=jnp.repeat(10e-19, len(z_guess)),
   ub=jnp.repeat(10e19, len(z_guess))
)
nlp.add_option('mu_strategy', 'adaptive')
nlp.add_option('tol', 1e-3)
nlp.add_option('max_iter',100)
x, info= nlp.solve(z_guess)


with open(f'solution_{sys.argv[1]}.file', 'wb') as f:
    pickle.dump(x, f,protocol = pickle.HIGHEST_PROTOCOL)

aggregate = Aggregate_plot(olg, t_0 = 0, t_1 = 100, name = 'test')
aggregate.create(alpha=.5, linestyle='dashed')
olg = update_optimal_trajectory(olg, x, t_0=1, t_1=t_1)
aggregate.update(alpha = .5)
aggregate.fig.savefig(f'aggregates_{sys.argv[1]}steps.jpg')
