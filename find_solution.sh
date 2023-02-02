# python find_solution.py baseline -a o -t 1 -T 200 -nis 20 -nit 20 -v 1;
# python find_solution.py baseline -in baseline -a c -t 1 -T 200 -nis 20 -nit 20 -v 1;
python find_solution.py reform -in baseline -a c -t 5 -T 200 -nis 10 -nit 20 -r reform -v 0&
python find_solution.py reform_delayed -in baseline -a c -t 5 -T 200 -nis 10 -nit 20 -r reform_delayed -v 0&
python find_solution.py private -in baseline -a c -t 5 -T 200 -nis 10 -nit 20 -r private -v 1&
python find_solution.py baseline_c -in baseline -a c -t 5 -T 200 -nis 10 -nit 20 -v 0|
tee out.txt
