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


olg = OLG_model(G=60,T=250,N=N,epsilon=epsilon, rho=rho,
                gov_ratio=0.2,gov_const=gov_const,
                sigma = sigma,Pi=Pi,r = r,price_M =price_M,
                price_E=price_E, tau_I=tau_I,tau_II=tau_II,tau_Ins=tau_Ins,
                tau_pi=tau_pi, tau_VA=tau_VA, tau_rho=tau_rho, beta = 0.99,
                theta =1,
                phi =0.23, # цель 38 миллионов в стеди стейт или 65% в начальном положении
                psi = 24.
                , omega=0.269
                , alpha = 0.35, delta=0.0608,
                A=A,initial=initial, Oil=Oil,eta =0.25,steady_max_iter=5000,max_iter=5000,
                steady_guess=steady_guess)


niter = 30
tol = 10e-5
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in range(niter):
        olg.steady_state()
        if (len(olg.steady_path)>1) and \
            (max(abs(olg.steady_path[-1][1] - olg.steady_path[-2][1]))<tol):
                break
                
                
with open(f'olg_{name}.file', 'wb') as f:
    pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)
    
olg.update_a_initial()
olg.create_guess(t_0=1,steady_start = 100)


fig, ax = plt.subplots(3,3, figsize = (16,16))
ax[0,0].plot(olg.k[0,:(olg.T+50)], alpha=0.5, color="black", label = r"$k_N$")
ax[0,1].plot(olg.Consumption[:(olg.T+50)], alpha=0.5, color="black", label = r"$C$")
ax[0,2].plot(olg.Labor[:(olg.T+50)], alpha=0.5, color="black", label = r"$L$")
ax[1,0].plot(olg.w[:(olg.T+50)], alpha=0.5, color="black", label = r"$w$")
ax[1,1].plot(olg.price[:(olg.T+50)], alpha=0.5, color="red", label = r"$p$")
ax[1,2].plot(olg.price[:(olg.T+50)], alpha=0.5, color="blue", label = r"$p_N$")
ax[2,0].plot(olg.c[0,:,1:(olg.T)].diagonal(-olg.G+1), alpha=0.5, color="black", label = r"$c_{f}$")
ax[2,0].plot(olg.c[1,:,1:(olg.T)].diagonal(-olg.G+1), alpha=0.5, color="red", label = r"$c_{f}$")
ax[2,1].plot(olg.l[0,:,1:(olg.T)].diagonal(-olg.G+1), alpha=0.5, color="black", label = r"$l_{f}$")
ax[2,1].plot(olg.l[1,:,1:(olg.T)].diagonal(-olg.G+1), alpha=0.5, color="red", label = r"$l_{m}$")
ax[2,2].plot(olg.a[0,:,1:(olg.T)].diagonal(-olg.G+1), alpha=0.5, color="black", label = r"$a_{f}$")
ax[2,2].plot(olg.a[1,:,1:(olg.T)].diagonal(-olg.G+1), alpha=0.5, color="red", label = r"$a_{m}$")


for row in ax:
    for col in row:
        col.legend()

msg = bot.send_plot(fig)
bot.clean_tmp_dir()

niter = 100
for i in range(niter):
    for t in range(olg.T):
        olg.update_government(t)
    for t in range(1, olg.T):
        olg.update_household(t)
    for t in tqdm(reversed(range(1,olg.T)),
                      desc = f'{name} path to steady {i}',
                      total=olg.T-1,
                          token=os.getenv('comp_bot_token'),
                          chat_id=os.getenv('chat_id')):

        olg.update_guess(t)
        
        
    ax[0,0].plot(olg.k[0,:(olg.T+50)], alpha = (i+1)/niter, color="black", label = r"$k_N$")
    ax[0,0].plot(olg.k[1,:(olg.T+50)], alpha = (i+1)/niter, color="red", label = r"$k_N$")
    ax[0,1].plot(olg.Consumption[:(olg.T+50)], alpha = (i+1)/niter, color="black", label = r"$C$")
    ax[0,2].plot(olg.Labor[:(olg.T+50)], alpha = (i+1)/niter, color="black", label = r"$L$")
    ax[1,0].plot(olg.w[:(olg.T+50)], alpha = (i+1)/niter, color="black", label = r"$w$")
    ax[1,1].plot(olg.price[:(olg.T+50)], alpha = (i+1)/niter, color="red", label = r"$p$")
    ax[1,2].plot(olg.price[:(olg.T+50)], alpha = (i+1)/niter, color="blue", label = r"$p_N$")
    ax[2,0].plot(olg.c[0,:,1:(olg.T)].diagonal(-olg.G+1), alpha = (i+1)/niter, color="black", label = r"$c_{f}$")
    ax[2,0].plot(olg.c[1,:,1:(olg.T)].diagonal(-olg.G+1), alpha = (i+1)/niter, color="red", label = r"$c_{f}$")
    ax[2,1].plot(olg.l[0,:,1:(olg.T)].diagonal(-olg.G+1), alpha = (i+1)/niter, color="black", label = r"$l_{f}$")
    ax[2,1].plot(olg.l[1,:,1:(olg.T)].diagonal(-olg.G+1), alpha = (i+1)/niter, color="red", label = r"$l_{m}$")
    ax[2,2].plot(olg.a[0,:,1:(olg.T)].diagonal(-olg.G+1), alpha = (i+1)/niter, color="black", label = r"$a_{f}$")
    ax[2,2].plot(olg.a[1,:,1:(olg.T)].diagonal(-olg.G+1), alpha = (i+1)/niter, color="red", label = r"$a_{m}$")
    bot.update_plot(msg, fig)
    with open(f'olg_{name}.file', 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)

    price_update = np.array([item[-1]['x'][-1] for key, item in olg.history.items() if len(item)!=0])
    error = abs(0.5*(olg.price[1:(olg.T)]-price_update)/(olg.price[1:(olg.T)]+price_update))
    print(error)
    if max(error) < tol/10:
        break
bot.clean_tmp_dir()

with open(f'olg_{name}.file', 'wb') as f:
    pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)