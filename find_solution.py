from olg import *
bot = TelegramBot(os.getenv('comp_bot_token'),
                      os.getenv('chat_id'))

T = int(sys.argv[1])
name = str(sys.argv[2])

pb_iteration = 0
progress_bar = tqdm(desc = f'{name} {pb_iteration}',
                      total=None,
                          token=os.getenv('comp_bot_token'),
                          chat_id=os.getenv('chat_id')) 


olg = OLG_model(G=60,T=T,N=N,epsilon=epsilon, rho=rho,
                gov_ratio=0.2,
                gov_const=gov_const,
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

aggregate = Aggregate_plot(olg, t_0 = 0, t_1=olg.T)
household = Household_plot(olg, g_0=30, g_1=60)

niter = 30
tol = 10e-5
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    for _ in range(niter):
        
        olg.steady_state()
         
        
with open(f'olg_{name}.file', 'wb') as f:
    pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)
    
olg.update_a_initial()
olg.create_guess(t_0=1,steady_start = 100)

aggregate.create(alpha=.5, linestyle='dashed')
household.create(alpha=.5, linestyle='dashed')

msg_aggregate, msg_household = bot.send_plot(aggregate.fig), bot.send_plot(household.fig)
bot.clean_tmp_dir()

niter = 100
for i in range(niter):
    for t in range(olg.T):
        gov_const = olg.gov_const
        olg.update_government(t)
        if gov_const != olg.gov_const:
            for _ in range(30):
                olg.steady()
            olg.update_a_initial()
            olg.create_guess(t_0=olg.T-100,steady_start = olg.T-50)
        olg.update_government(t)
    for t in range(1, olg.T):
        olg.update_household(t)
    progress_bar.refresh(nolock=True)
    pb_iteration += pb_iteration
    for t in range(1, olg.T):
        olg.update_guess(t)
        progress_bar.update(1)

        
    aggregate.update(alpha=.1, linestyle='solid')
    household.update(alpha=.5, linestyle='solid')
        
    bot.update_plot(msg_aggregate, aggregate.fig)
    bot.update_plot(msg_household, household.fig)
    
    with open(f'olg_{name}.file', 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)