python find_solution.py baseline_mid_demo_mid_oil_exo -in baseline_mid_demo_mid_oil_exo -a c -t 1 -T 200 -u exogenous_labor -nis 20 -nit 100 -v 1 -ds medium -os mid&
python find_solution.py baseline_mid_demo_high_oil_exo -in baseline_mid_demo_high_oil_exo  -a c -t 1 -T 200 -u exogenous_labor -nis 20 -nit 100 -v 1 -ds medium -os high&
python find_solution.py baseline_mid_demo_price_cap_oil_exo -in baseline_mid_demo_price_cap_oil_exo -a c -t 1 -T 200 -u exogenous_labor -nis 20 -nit 100 -v 1 -ds medium -os price_cap;
# python find_solution.py baseline_high_demo_price_cap_oil_exo -in baseline_high_demo_price_cap_oil_exo -a c -t 1 -T 200 -u exogenous_labor -nis 20 -nit 100 -v 1 -ds high -os price_cap;

# python find_solution.py baseline -in baseline -a c -t 1 -T 200 -nis 20 -nit 20 -v 1;
# python find_solution.py reform -in baseline -a c -t 5 -T 200 -nis 10 -nit 20 -r reform -v 0&
# python find_solution.py reform_delayed -in baseline -a c -t 5 -T 200 -nis 10 -nit 20 -r reform_delayed -v 0&
# python find_solution.py private -in baseline -a c -t 5 -T 200 -nis 10 -nit 20 -r private -v 1&
# python find_solution.py baseline_c -in baseline -a c -t 5 -T 200 -nis 10 -nit 20 -v 0|
# tee out.txt
