from olg import *
name = str(sys.argv[1])
gov_strategy = str(sys.argv[2])
gov_retirement_strategy = str(sys.argv[3])
niter_steady=30
niter_transition=100

progress_bar = tqdm(desc = f'{name} {pb_iteration}',
                      total=None,
                          token=os.getenv('comp_bot_token'),
                          chat_id=os.getenv('chat_id')) 
olg = OLG_model(gov_strategy=gov_strategy,
                gov_retirement_strategy=gov_retirement_strategy)

aggregate = Aggregate_plot(olg, t_0 = 0, t_1=olg.T)
household = Household_plot(olg, g_0=30, g_1=60)
government = Gov_plot(olg, t_0 = 0, t_1 = olg.T)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    for _ in range(niter_steady):

        olg.steady_state()

with open(f'result/{name}.file', 'wb') as f:
    pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)

olg.update_a_initial()
olg.create_guess(t_0=1,steady_start = 100)

aggregate.create(alpha=.5, linestyle='dashed')
household.create(alpha=.5, linestyle='dashed')
government.create(alpha=.5, linestyle='dashed')

msg_aggregate, msg_household, msg_government = bot.send_plot(aggregate.fig), bot.send_plot(household.fig), bot.send_plot(government.fig)

bot.clean_tmp_dir()

t_0=1
for i in range(niter_transition):
    for t in range(olg.T):
        gov_ratio = olg.gov_ratio[0]
        olg.update_government(t)

        if olg.gov_adaptation_time is not None:
                for _ in range(niter_steady):
                    olg.steady_state()
                    olg.create_guess(t_0=olg.gov_adaptation_time,
                                     steady_start = olg.T-50)
        if gov_ratio != olg.gov_ratio[0]:
            for _ in range(niter_steady):
                olg.steady_state()
            olg.update_a_initial()
            olg.create_guess(t_0=olg.T-100,steady_start = olg.T-50)
        olg.update_government(t)
    for t in range(t_0, olg.T):
        olg.update_household(t, t)
    progress_bar.refresh(nolock=True)
    pb_iteration += pb_iteration
    for t in range(t_0, olg.T):
        olg.update_guess(t)
        progress_bar.update(1)


    aggregate.update(alpha=.1, linestyle='solid')
    household.update(alpha=.1, linestyle='solid')
    government.update(alpha=.1, linestyle='solid')


    bot.update_plot(msg_aggregate, aggregate.fig)
    bot.update_plot(msg_household, household.fig)
    bot.update_plot(msg_government, government.fig)
    with open(f'result/olg_{name}.file', 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)