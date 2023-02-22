import numpy as np




# Globals
G_TOTAL = int(os.getenv('G_TOTAL'))
G_MODEL = int(os.getenv('G_MODEL'))
MAX_TIME = int(os.getenv('MAX_TIME'))
STEADY_TIME = int(os.getenv('STEADY_TIME'))

# Precomputed blocks
demography = np.load(os.path.join('assets', f'demography_{demo_scenario}.pickle'),allow_pickle='TRUE').item()

oil = np.load(os.path.join('assets', f'oil_{oil_scenario}.pickle'),allow_pickle='TRUE').item()

# Households
phi = np.array([1, 1.28])
upsilon = 5.
beta = 0.96
iota = np.array([0.8, 0.8])
N, Pi, epsilon, rho, rho_reform = demography['population'], demography['survival_probability'], demography['epsilon'], demography['rho'], demography['rho_reform']

# World
r = np.repeat(0.04, MAX_TIME)
price_M = np.repeat(1., MAX_TIME)
price_E = np.repeat(1., MAX_TIME)

# Oil 
Y_O , price_O, psi_O= oil['Y_O'], oil['price_O'], oil['psi_O']

# Production
A_N = np.cumprod(np.concatenate(([1.],np.linspace(1.02,1.01,99), np.array([1.01 for _ in range(MAX_TIME-100)]))))
A_E = np.cumprod(np.concatenate(([1.],np.linspace(1.02,1.01,99), np.array([1.01 for _ in range(MAX_TIME-100)]))))
A = np.array([A_N, A_E])

omega = 0.289
alpha = 0.35
psi = 1.5
delta = 0.1
GDP_initial = 103861.7
K_initial = (GDP_initial - Y_O[0])*alpha / (r[0] + delta) # 205903
K_E_initial = K_initial*0.168 # 34592 
K_N_initial = K_initial*(1-0.168) # 171312

I_initial = 22764.5
I_N_initial = I_initial * (1-0.168)
I_E_initial = I_initial * 0.168

# Population scaling
L_initial = 1/(K**0.35/ Y_non_oil)**(1/0.65)
L_unscaled = sum((phi[s] * N[s, :, 0] * epsilon[s, :, 0]).sum() for s in range(2))
N = N * (L_initial / L_unscaled)
L_N_initial = L_initial * (1-0.168)
L_E_initial = L_initial * (0.168)





# Taxes
tau_I = np.repeat(0.13, MAX_TIME)
tau_II = np.repeat(0., MAX_TIME)
tau_Ins = np.repeat(0.08, MAX_TIME)
tau_pi = np.repeat(0.2, MAX_TIME)
tau_VA = np.repeat(0.2, MAX_TIME)
tau_rho = np.repeat(0.22, MAX_TIME)
tau_O = np.repeat(0.78, MAX_TIME)
tax_LS = np.repeat(0., MAX_TIME)*A_N




# Government
sigma = np.array([np.repeat(0.293, MAX_TIME), np.repeat(0.33, MAX_TIME)])
Gov_initial = 18394.
Debt_initial = -20707.7
target_debt_to_gdp = 0.3
tax_sensitivity = {"VA_lag":0., "VA":1.8, "I":0., "I_squared":0.}
Deficit_initial = 3035.6


# Computation
eta =0.25
steady_max_iter=5000
max_iter=500
initial = {"price_N":1.,
         "K_N":K_N_initial,
         "L_N":L_N_initial,
         "I_N":I_N_initial,
         "K_E":K_E_initial,
         "L_E":L_E_initial,
         "I_E":I_E_initial,
         "Debt":Debt_initial,
         "Gov":Gov_initial, 
         "Deficit":Deficit_initial,
           "lmbda_E":0.5,
           "lmbda_N":0.5
}
           
steady_guess = np.array([2.28699006e+00, 9.74048999e-01, 6.09327218e-02, 1.04489377e+01,
       1.00000000e+00, 1.00000000e+00, 6.74383209e+02, 4.83939191e+01,
       6.17294122e+03])
# steady_guess = np.array([ 1.57841123e+00,  6.55377948e-01,  7.87994284e-01,  1.03943485e+01,
#         1.00000000e+00,  1.00000000e+00,  6.25792410e+02,  6.68536161e+01,
#        -6.55468099e+02])
#  k_N, l_N, k_E, w, price, price_N, Consumption,  Labor, Assets
