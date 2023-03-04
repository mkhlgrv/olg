
import argparse
from dotenv import load_dotenv
import os
from os.path import exists

load_dotenv()
import argparse

parser = argparse.ArgumentParser(description='Find solution of olg model')
parser.add_argument('output_name',  type=str, help = "output file name")
parser.add_argument('-in','--input_name',  type=str, help = "input file name", default = "")
parser.add_argument('-a','--action',  type=str, help = "c for continue, o for overwrite", default = "c")
parser.add_argument('-t','--t_0',  type=int, help = "starting time", default = 1)
parser.add_argument('-T','--T',  type=int, help = "steady time", default = 200)
parser.add_argument('-u','--utility',  type=str, help = "utility function type, hybrid or exogenous_labor", default = 'exogenous_labor')
parser.add_argument('-r', '--reform',  type=str, help = "retirement system type", default = "")
parser.add_argument('-nis','--niter_steady',  type=int, help = "Steady state niter", default = 1)
parser.add_argument('-nit','--niter_transition',  type=int, help = "Transition path niter", default = 10)
parser.add_argument('-v','--verbose',  type=int, help = "if 0, bot sends only last plot", default = 1)
parser.add_argument('-ds','--demo_scenario',  type=str, help = "medium, low, or high", default = 'medium')
parser.add_argument('-os','--oil_scenario',  type=str, help = "high, low, mid, price_cap", default = 'mid')





args = parser.parse_args()
args.verbose = bool(args.verbose)
print(args.verbose)

os.environ["demo_scenario"] = args.demo_scenario
os.environ["oil_scenario"] = args.oil_scenario

from olg import *
    
file_name = f'result/{args.output_name}.file'   
input_filename = f'result/{args.input_name}.file'


progress_bar = tqdm(desc = f'{args.output_name} {pb_iteration}',
                      total=None,
                          token=os.getenv('comp_bot_token'),
                          chat_id=os.getenv('chat_id')) 
if exists(input_filename) and args.action == 'c' :
    
    with open(input_filename, 'rb') as f:
        olg = pickle.load(f)
        # olg = OLG_model(T=args.T)
        # olg.copy(input_model)
        # olg = input_model
        if args.reform != '':
            if args.reform == "reform":
                olg.rho[:,:,args.t_0:] = rho_reform[:,:,args.t_0:]
                   
            if args.reform == "reform_delayed":
                olg.rho[:,:,args.t_0:] = rho_reform_delayed[:,:,args.t_0:]
            
            if args.reform == "private":
                olg.rho[:,:,args.t_0:] = rho_private[:,:,args.t_0:]
                olg.tau_rho[args.t_0:] = tau_rho_private[args.t_0:]
                olg.rho_lamp_sum_coef[args.t_0:] = rho_lamp_sum_coef_private[args.t_0:]
                
            for _ in range(args.niter_steady):
                olg.update_steady_state()
            olg.create_guess(t_0=olg.T-50,steady_start = olg.T-25)
            
elif args.action == 'o':
    olg = OLG_model(T=args.T, utility=args.utility)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for _ in range(args.niter_steady):

            olg.update_steady_state()
            
    with open(file_name, 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)
        
    olg.update_a_initial()
    olg.create_guess(t_0=0,steady_start = olg.T-25)
else:
    print('unsupported action')

    
aggregate = Aggregate_plot(olg, t_0 = 0, t_1=olg.T, name = args.output_name)
household = Household_plot(olg, g_0=30, g_1=150, name = args.output_name)
government = Gov_plot(olg, t_0 = 0, t_1 = olg.T, name = args.output_name)

aggregate.create(alpha=.5, linestyle='dashed')
household.create(alpha=.5, linestyle='dashed')
government.create(alpha=.5, linestyle='dashed')

if args.verbose:
    
    msg_aggregate, msg_household, msg_government = send_figure(aggregate), send_figure(household),send_figure(government)
    
for _ in range(20):
    
    for t in range(args.t_0,olg.T):
        olg.update_government(t, 1)
    olg.tau_VA[olg.T:] = olg.tau_VA[olg.T-1] 

    if args.verbose:
        government.update(alpha=.1, linestyle='solid')
        send_figure(government, msg_government)
    
    for t in range(args.t_0, olg.T):
            olg.update_household(t, t)
    if args.verbose:
        household.update(alpha=.1, linestyle='solid')
        send_figure(household, msg_household)
    for _ in range(args.niter_steady):
        olg.update_steady_state()
    olg.update_a_initial()
    olg.create_guess(t_0=0,steady_start = olg.T-25)
    if args.verbose:
        aggregate.update(alpha=.1, linestyle='solid')
        send_figure(aggregate, msg_aggregate)
    


for i in range(args.niter_transition):
    for t in range(args.t_0,olg.T-25):
         olg.update_government(t, 1)
    olg.tau_VA[(olg.T-25):] = olg.tau_VA[olg.T-26] 

    government.update(alpha=.1, linestyle='solid')
    if args.verbose:
        send_figure(government, msg_government)

    for t in range(args.t_0, olg.T-25):
         olg.update_household(t, t)

    household.update(alpha=.1, linestyle='solid')
    if args.verbose:
         send_figure(household, msg_household)

    progress_bar.refresh(nolock=True)
    pb_iteration += pb_iteration
    for t in range(args.t_0, olg.T-25):
        olg.update_guess(t)
        progress_bar.update(1)
    olg.update_capital_guess(args.t_0, olg.T-25)
    aggregate.update(alpha=.1, linestyle='solid')
    if args.verbose:
        send_figure(aggregate, msg_aggregate)

   # for _ in range(args.niter_steady):
   #     olg.update_steady_state()
   # olg.create_guess(t_0=olg.T-50,steady_start = olg.T-25)

    with open(file_name, 'wb') as f:
        pickle.dump(olg, f,protocol = pickle.HIGHEST_PROTOCOL)

if not args.verbose:
    send_figure(aggregate), send_figure(household),send_figure(government)
