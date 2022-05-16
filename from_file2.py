from olg import *
bot = TelegramBot(os.getenv('comp_bot_token'),
                      os.getenv('chat_id'))
import sys
import pickle
T = int(sys.argv[1])
if sys.argv[2] is not None:
    name = str(sys.argv[2])
else:
    name = ""

olg = OLG_model(G=60,T=T,N=N,epsilon=epsilon, rho=rho,
                gov_ratio=0.3195,
                sigma = sigma,Pi=Pi,r = r,price_M =price_M,
                price_E=price_E, tau_I=tau_I,tau_II=tau_II,tau_Ins=tau_Ins,
                tau_pi=tau_pi, tau_VA=tau_VA, tau_rho=tau_rho, beta = 0.99,
                theta =1,
                phi =0.23, # цель 38 миллионов в стеди стейт или 65% в начальном положении
                psi = 5.
                , omega=0.269
                , alpha = 0.35, delta=0.0608,
                A=A,initial=initial, Oil=Oil,eta =0.25,steady_max_iter=5000,max_iter=5000,
                steady_guess=steady_guess)
    
with open('olg_base.file', 'rb') as f:
    olg_old=pickle.load(f)
    
olg.steady = olg_old.steady
olg.k, olg.K, olg.i, olg.K, olg.Labor, olg.w, olg.price_N, olg.I, olg.L, olg.lmbda, olg.price, olg.L_share, olg.c, olg.l, olg.a, olg.Assets = olg_old.k, olg_old.K, olg_old.i, olg_old.K, olg_old.Labor, olg_old.w, olg_old.price_N, olg_old.I, olg_old.L, olg_old.lmbda, olg_old.price, olg_old.L_share, olg_old.c, olg_old.l, olg_old.a, olg_old.Assets



olg.update_a_initial()
olg.create_guess()


fig, ax = plt.subplots(2,2, figsize = (12,12))
ax[0,0].plot(olg.price[:(olg.T+50)], alpha=0.05, color="black", label = "p")
ax[0,0].plot(olg.w[:(olg.T+50)], alpha=0.05, color="red", label = "w")
ax[0,1].plot(olg.Consumption[:(olg.T+50)], alpha=0.05, color="black", label = "C")
ax[1,0].plot(olg.Labor[:(olg.T+50)], alpha=0.05, color="black", label = "Labor")
ax[1,1].plot(olg.k[0,:(olg.T+50)], alpha=0.05, color="black", label = "k_N")
for row in ax:
    for col in row:
        col.legend()

msg = bot.send_plot(fig)
bot.clean_tmp_dir()

niter = 20
tol=1e-4
eta = olg.eta
olg.eta=0.8
for i in range(niter):
    olg.evaluate_initial_state()  
    for t in range(1, olg.T):
        olg.update_household(t)
        olg.update_government(t)
  
    for t in tqdm(reversed(range(1,olg.T)),
                      desc = f'{name} path to steady {i}',
                      total=olg.T-1,
                          token=os.getenv('comp_bot_token'),
                          chat_id=os.getenv('chat_id')):
        print(olg.price[t])
        olg.update_guess(t)
        print(olg.price[t])
        olg.eta=eta
    olg.last_guess = None
        
        
    ax[0,0].plot(olg.price[:(olg.T+50)], alpha=i/niter, color="black")
    ax[0,0].plot(olg.w[:(olg.T+50)], alpha=i/niter, color="red")
    ax[0,1].plot(olg.Consumption[:(olg.T+50)], alpha=i/niter, color="black")
    ax[1,0].plot(olg.Labor[:(olg.T+50)], alpha=i/niter, color="black")
    ax[1,1].plot(olg.k[0,:(olg.T+50)], alpha=i/niter, color="black")
    bot.update_plot(msg, fig)
    with open(f'olg_{name}.file', 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)

    price_update = np.array([item[-1]['x'][-1] for key, item in olg.history.items() if len(item)!=0])
    error = abs(0.5*(olg.price[1:(olg.T)]-price_update)/(olg.price[1:(olg.T)]+price_update))
    print(error)
    if max(error) < tol/10:
        break
bot.clean_tmp_dir()