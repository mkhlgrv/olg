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

with open('olg_base.file', 'rb') as f:
    olg=pickle.load(f)

    
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
    for t in range(1,olg.T):
        olg.evaluate_initial_state()
        olg.update_household(t)
        olg.update_government(t)
    
    for t in tqdm(reversed(range(1,olg.T)),
                      desc = f'{name} path to steady {i}',
                      total=olg.T-1,
                          token=os.getenv('comp_bot_token'),
                          chat_id=os.getenv('chat_id')):
        olg.update_guess(t)
        
        
    ax[0,0].plot(olg.price[:(olg.T+50)], alpha=i/niter, color="black")
    ax[0,0].plot(olg.w[:(olg.T+50)], alpha=i/niter, color="red")
    ax[0,1].plot(olg.Consumption[:(olg.T+50)], alpha=i/niter, color="black")
    ax[1,0].plot(olg.Labor[:(olg.T+50)], alpha=i/niter, color="black")
    ax[1,1].plot(olg.k[0,:(olg.T+50)], alpha=i/niter, color="black")
    bot.update_plot(msg, fig)
    with open(f'olg_{name}.file', 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)

    price_update = np.array([item[-1]['x'][-1] for key, item in olg.history.items() if len(item)!=0 and key >= t_0])
    error = abs(0.5*(olg.price[t_0:(olg.T)]-price_update)/(olg.price[t_0:(olg.T)]+price_update))
    print(error)
    if max(error) < tol/10:
        break
bot.clean_tmp_dir()

with open(f'olg_{name}_reform.file', 'wb') as f:
    pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)