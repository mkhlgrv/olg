import numpy as np
demography = np.load('demography.npy',allow_pickle='TRUE').item()
n_generations = 60


max_time = demography["N"][0].shape[1]
sigma = np.array([0.356 for _ in range(max_time)])
sigma_low = np.array([0.3 for _ in range(max_time)])
r = np.array([0.078 for _ in range(max_time)])
price_M = np.array([1. for _ in range(max_time)])
price_E = np.array([1. for _ in range(max_time)])
tau_I = np.array([0.13 for _ in range(max_time)])
tau_II = np.array([0. for _ in range(max_time)])
tau_Ins = np.array([0.08 for _ in range(max_time)])
tau_pi = np.array([0.2 for _ in range(max_time)])
tau_VA = np.array([0.18 for _ in range(max_time)])
tau_rho = np.array([0.22 for _ in range(max_time)])
A_N = np.cumprod(np.concatenate(([1.],np.linspace(1.02,1.01,99), np.array([1.01 for _ in range(max_time-100)]))))
A_E = np.cumprod(np.concatenate(([1.],np.linspace(1.02,1.01,99), np.array([1.01 for _ in range(max_time-100)]))))


N = demography["N"]
Pi = demography["Pi"]
epsilon = demography["epsilon"]
rho = demography["rho"]
rho_reform = demography["rho_reform"]
rho_reform_delayed = demography["rho_reform_delayed"]

A = np.array([A_N, A_E])

gov_ratio=0.2
GDP_initial = 44.089**0.35 * 72.508**0.65
Oil_initial = GDP_initial*0.0914 /(1-0.0914)
Debt_initial = GDP_initial*0.10051
Oil = np.array([Oil_initial for _ in range(max_time)])
I_init = 5.202
Consumption_init = GDP_initial*(1-gov_ratio) + Oil_initial - I_init
deficit_ratio_initial = 0.01069

initial = {"a_initial_sum":500,
                "price_N":1.,
                 "K_N":44.089*(1-0.198),
                 "L_N":72.508*(1-0.198),
                 "I_N":I_init*(1-0.198),
                 "K_E":44.089*(0.198),
                 "L_E":72.508*(0.198),
                 "Debt":Debt_initial,
                 "Gov":GDP_initial*gov_ratio, 
                 "I_E":I_init*(0.198),
                  "C":Consumption_init}


           
steady_guess = np.array([ 2.46949299,  0.98187727,  0.04557998, 17.60330632,  1.        ,
          1.        ])