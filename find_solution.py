from olg import *
bot = TelegramBot(os.getenv('comp_bot_token'),
                      os.getenv('chat_id'))
import sys
import pickle
T = int(sys.argv[1])


olg = OLG_model(G=60,T=T,N=N,epsilon=epsilon, rho=rho,
                sigma = sigma,Pi=Pi,r = r,price_M =price_M,
                price_E=price_E, tau_I=tau_I,tau_II=tau_II,tau_Ins=tau_Ins,
                tau_pi=tau_pi, tau_VA=tau_VA, tau_rho=tau_rho, beta = 0.995,
                theta =1,
                phi =0.3, # цель 38 миллионов в стеди стейт или 65% в начальном положении
                psi = 163.
                , omega=0.269
                , alpha = 0.35, delta=0.0608,
                A=A,initial=initial, eta =0.25,steady_max_iter=5000,max_iter=5000,
                steady_guess=steady_guess)


niter = 30
tol = 10e-5
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for _ in tqdm(range(niter),
                  desc = 'Steady state',
                      token=os.getenv('comp_bot_token'),
                      chat_id=os.getenv('chat_id')):
        olg.steady_state()
        if (len(olg.steady_path)>1) and \
            (max(abs(olg.steady_path[-1][1] - olg.steady_path[-2][1]))<tol):
                break
                
                
with open('olg_result.file', 'wb') as f:
    pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)
    
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
for i in range(niter):
    
    
    for t in tqdm(reversed(range(1,olg.T)),
                      desc = f'path to steady {i}',
                      total=olg.T-1,
                          token=os.getenv('comp_bot_token'),
                          chat_id=os.getenv('chat_id')):
        olg.update_guess(t)
        
    olg.evaluate_initial_state()
    ax[0,0].plot(olg.price[:(olg.T+50)], alpha=i/niter, color="black")
    ax[0,0].plot(olg.w[:(olg.T+50)], alpha=i/niter, color="red")
    ax[0,1].plot(olg.Consumption[:(olg.T+50)], alpha=i/niter, color="black")
    ax[1,0].plot(olg.Labor[:(olg.T+50)], alpha=i/niter, color="black")
    ax[1,1].plot(olg.k[0,:(olg.T+50)], alpha=i/niter, color="black")
    bot.update_plot(msg, fig)

    price_update = np.array([item[-1]['x'][-1] for key, item in olg.history.items() if len(item)!=0])
    error = abs(0.5*(olg.price[1:(olg.T)]-price_update)/(olg.price[1:(olg.T)]+price_update))
    print(error)
    if max(error) < tol/10:
        break
bot.clean_tmp_dir()

with open('olg_result.file', 'wb') as f:
    pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)