from olg import *
bot = TelegramBot(os.getenv('comp_bot_token'),
                      os.getenv('chat_id'))

name = str(sys.argv[1])

pb_iteration = 0
progress_bar = tqdm(desc = f'{name} {pb_iteration}',
                      total=None,
                          token=os.getenv('comp_bot_token'),
                          chat_id=os.getenv('chat_id')) 

with open(f'olg_{name}.file', 'rb') as f:
    olg = pickle.load(f)
    
aggregate = Aggregate_plot(olg, t_0 = 0, t_1=olg.T)
household = Household_plot(olg, g_0=30, g_1=60)
         
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
            for _ in range(10):
                olg.steady_state()
            olg.update_a_initial()
            olg.create_guess(t_0=olg.T-100,steady_start = olg.T-50)
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