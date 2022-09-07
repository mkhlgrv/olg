import numpy as np
demography = np.load('demography.npy',allow_pickle='TRUE').item()
n_generations = 60


max_time = demography["N"][0].shape[1]
# sigma = np.array([0.356 for _ in range(max_time)])
sigma = np.array([0.3 for _ in range(max_time)])
r = np.array([0.06 for _ in range(max_time)])
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
rho_private = demography["rho_private"]
tau_rho_private = np.array([0.22 for _ in range(max_time)])
tau_rho_private[15:65] = np.linspace(tau_rho_private[15],0, 50, endpoint=False)
tau_rho_private[65:] = 0
rho_lamp_sum_coef = np.array([1. for _ in range(max_time)])

rho_lamp_sum_coef_private = np.array([1. for _ in range(max_time)])
rho_lamp_sum_coef_private[15:65] = np.linspace(1.,0, 50, endpoint=False)
rho_lamp_sum_coef_private[65:] = 0

A = np.array([A_N, A_E])

K_initial = 207.875
Y_initial = K_initial**0.35 * 72.508**0.65
Rho_lamp_sum_initial = 3.
Oil_initial = 0.083*Y_initial
oil_price = np.array([0.99**i for i in range(16)]+[0.99**16 for _ in range(max_time-16)])
GDP_initial = Y_initial+Oil_initial
Gov_initial = (0.18*GDP_initial+Rho_lamp_sum_initial)# 21.#12.184183598971048#+Rho_lamp_init # скорее 21 должно быть
Debt_initial = 0.10051*GDP_initial
I_initial = 0.215*GDP_initial
Oil = np.array([Oil_initial*(1-0.012)**(i) for i in range(max_time)])*oil_price
deficit_ratio_initial = 0.01069

initial = {"lamp_sum_tax":0.17 # калибруем так, чтобы уловить дефицит в 2014
                ,"price_N":1.,
                 "K_N":K_initial*(1-0.198),
                 "L_N":72.508*(1-0.198),
                 "I_N":I_initial*(1-0.198),
                 "K_E":K_initial*(0.198),
                 "L_E":72.508*(0.198),
                 "Debt":Debt_initial,
                 "Gov":Gov_initial, 
                 "I_E":I_initial*(0.198),
                 "Rho_lamp_sum":Rho_lamp_sum_initial}
           
steady_guess = np.array([2.3167618 ,   0.72691265,   0.87036366,  11.62847021,
         1.        ,   1.        , 447.5913557 ,  48.56260336,
       510.48451492])
