from olg import *
bot = TelegramBot(os.getenv('comp_bot_token'),
                      os.getenv('chat_id'))

source_name = str(sys.argv[1])
name = str(sys.argv[2])

pb_iteration = 0
progress_bar = tqdm(desc = f'{source_name} reform {pb_iteration}',
                      total=None,
                          token=os.getenv('comp_bot_token'),
                          chat_id=os.getenv('chat_id')) 


with open(f'olg_{source_name}.file', 'rb') as f:
    olg = pickle.load(f)
olg.rho = rho_reform_delayed

aggregate = Aggregate_plot(olg, t_0 = 0, t_1=olg.T)
household = Household_plot(olg, g_0=30, g_1=60)

niter = 30
tol = 10e-5
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    for _ in range(niter):
        olg.steady_state()
         
        
with open(f'olg_{source_name}_with_reform_{name}.file', 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)
t_0 = 15
olg.create_guess(t_0,steady_start = 100+t_0)

aggregate.create(alpha=.5, linestyle='dashed')
household.create(alpha=.5, linestyle='dashed')

msg_aggregate, msg_household = bot.send_plot(aggregate.fig), bot.send_plot(household.fig)
bot.clean_tmp_dir()

niter = 100
for i in range(niter):
    for t in range(t_0,olg.T):
        olg.update_government(t)
    for t in range(t_0, olg.T):
        olg.update_household(t)
    progress_bar.refresh(nolock=True)
    pb_iteration += pb_iteration
    for t in range(t_0, olg.T):
        olg.update_guess(t)
        progress_bar.update(1)

        
    aggregate.update(alpha=.1, linestyle='solid')
    household.update(alpha=.5, linestyle='solid')
        
    bot.update_plot(msg_aggregate, aggregate.fig)
    bot.update_plot(msg_household, household.fig)
    
    with open(f'olg_{source_name}_with_reform_{name}.file', 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)