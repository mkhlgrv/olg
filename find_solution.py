from olg import *
import argparse
from os.path import exists

name = sys.argv[1]
action = sys.argv[2]
input_name = sys.argv[3]
t_0 = int(sys.argv[4])
if input_name=="":
    input_name = name
    
file_name = f'result/{name}.file'   
input_filename = f'result/{input_name}.file'

keys = sys.argv[5::2]
values = sys.argv[6::2]
kwargs = {k:v for k, v in zip(keys, values)}

niter_steady=30

progress_bar = tqdm(desc = f'{name} {pb_iteration}',
                      total=None,
                          token=os.getenv('comp_bot_token'),
                          chat_id=os.getenv('chat_id')) 
if exists(input_filename) and action == 'c' :
    with open(input_filename, 'rb') as f:
        input_model = pickle.load(f)
        olg = OLG_model(**kwargs)
        olg.copy(input_model)
        if "rho" in kwargs:
            if kwargs["rho"] == "rho_reform":
                olg.rho = rho_reform
                   
            if kwargs["rho"] == "rho_reform_delayed":
                olg.rho = rho_reform_delayed
            
            if kwargs["rho"] == "private":
                olg.rho = rho_private
                olg.sigma = sigma_private
                
            for _ in range(niter_steady):
                olg.steady_state()
            olg.create_guess(t_0=olg.T-150,steady_start = olg.T-50)
        if "gov_retirement_strategy" in kwargs:
            if kwargs["gov_retirement_strategy"] == "fixed_tau_rho":
                olg.gov_retirement_strategy = "fixed_tau_rho"
                   
            if kwargs["gov_retirement_strategy"] == "fixed_sigma":
                olg.gov_retirement_strategy = "fixed_sigma"
            
                
            for _ in range(niter_steady):
                olg.steady_state()
            olg.create_guess(t_0=olg.T-150,steady_start = olg.T-50)
else:
    olg = OLG_model(**kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for _ in range(niter_steady):

            olg.steady_state()
    with open(file_name, 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)
    olg.update_a_initial()
    olg.create_guess(t_0=1,steady_start = 100)

# это удалить ---
# for _ in range(niter_steady):
#     olg.steady_state()
# olg.create_guess(t_0=30,steady_start = 200)
# for i in range(max_time):
#     olg.update_government(i,1)
# with open(file_name, 'wb') as f:
#     pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)
# ---
    
aggregate = Aggregate_plot(olg, t_0 = 0, t_1=olg.T)
household = Household_plot(olg, g_0=30, g_1=250)
government = Gov_plot(olg, t_0 = 0, t_1 = olg.T)

aggregate.create(alpha=.5, linestyle='dashed')
household.create(alpha=.5, linestyle='dashed')
government.create(alpha=.5, linestyle='dashed')

msg_aggregate, msg_household, msg_government = bot.send_plot(aggregate.fig), bot.send_plot(household.fig), bot.send_plot(government.fig)

bot.clean_tmp_dir()


niter_transition=50
for i in range(niter_transition):
    for t in range(t_0,olg.T):
        olg.update_government(t, 1)
        
    government.update(alpha=.1, linestyle='solid')
    bot.update_plot(msg_government, government.fig)
    
    for t in range(t_0, olg.T):
        olg.update_household(t, t)
    household.update(alpha=.1, linestyle='solid')
    bot.update_plot(msg_household, household.fig)  
    
    progress_bar.refresh(nolock=True)
    pb_iteration += pb_iteration
    for t in reversed(range(t_0, olg.T)):
        olg.update_guess(t)
        progress_bar.update(1)
    
    aggregate.update(alpha=.1, linestyle='solid')
    bot.update_plot(msg_aggregate, aggregate.fig)
    
    with open(file_name, 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)
