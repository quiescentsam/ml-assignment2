

import numpy as np
import mlrose_hiive
import pandas as pd
import seaborn as sns
save_path = 'flipflop'
import time
import matplotlib.pyplot as plt


#edges = [(0, 1), (1, 2), (0, 2), (1, 3), (2, 3), (3, 4)]
#fitness = mlrose_hiive.MaxKColor(edges)
#problem = mlrose_hiive.DiscreteOpt(length=5, fitness_fn=fitness, maximize=False, max_val=2)


fitness = mlrose_hiive.FlipFlop()
problem = mlrose_hiive.DiscreteOpt(length =500, fitness_fn = fitness, maximize = True, max_val = 2)



def maximum(a, b, c):
    if (a >= b) and (a >= c):
        largest = a

    elif (b >= a) and (b >= c):
        largest = b
    else:
        largest = c

    return largest

itersList = 2 ** np.arange(11)

rhc = mlrose_hiive.RHCRunner(problem=problem,
                       experiment_name="RCH",
                       output_directory="../flipflop_problem",
                       seed=None,
                       iteration_list=itersList,
                       max_attempts=1000,
                       restart_list=[0])


rhc_run_stats, rhc_run_curves = rhc.run()

plt.figure()
plt.title('RHC Runner for flipflop')
plt.plot(rhc_run_curves.Fitness, label='Fitness score',color="navy")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'rhc_fitness_flipflop.png')

best_state, best_fitness,fitness_curve = mlrose_hiive.random_hill_climb(problem,
			  max_attempts = 1000, max_iters = 8388608,
				#init_state = init_state,
				random_state = 0,curve=True)
plt.figure()
plt.title('RHC Runner for flipflop')
plt.plot(rhc_run_curves.Time, label='Time',color="navy")
plt.xlabel('Iteration')
plt.ylabel("Time")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'rhc_time_flipflop.png')



sa = mlrose_hiive.SARunner(problem=problem,
                     experiment_name="SA",
                     output_directory="../flipflop_problem",
                     seed=None,
                     iteration_list=itersList,
                     max_attempts=1000,
                     temperature_list=[1],
                     decay_list=[mlrose_hiive.ExpDecay])
sa_run_stats, sa_run_curves = sa.run()


sa = mlrose_hiive.SARunner(problem=problem,
                     experiment_name="SA",
                     output_directory="../flipflop_problem",
                     seed=None,
                     iteration_list=itersList,
                     max_attempts=1000,
                     temperature_list=[10],
                     decay_list=[mlrose_hiive.ExpDecay])
sa_run_stats1, sa_run_curves1 = sa.run()


sa = mlrose_hiive.SARunner(problem=problem,
                     experiment_name="SA_final",
                     output_directory="../flipflop_problem",
                     seed=None,
                     iteration_list=itersList,
                     max_attempts=1000,
                     temperature_list=[100],
                     decay_list=[mlrose_hiive.ExpDecay])
sa_run_stats2, sa_run_curves2 = sa.run()


sa = mlrose_hiive.SARunner(problem=problem,
                     experiment_name="SA_final",
                     output_directory="../flipflop_problem",
                     seed=None,
                     iteration_list=itersList,
                     max_attempts=1000,
                     temperature_list=[250],
                     decay_list=[mlrose_hiive.ExpDecay])
sa_run_stats3, sa_run_curves3 = sa.run()


plt.figure()
plt.title('SA Runner for flipflop')
plt.plot(sa_run_curves.Fitness, label='Fitness score temp = 1',color="navy")
plt.plot(sa_run_curves1.Fitness, label='Fitness score temp = 10',color="red")
plt.plot(sa_run_curves2.Fitness, label='Fitness score temp = 100',color="yellow")
plt.plot(sa_run_curves3.Fitness, label='Fitness score temp = 250',color="green")

plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'sa_fitness_flipflop.png')



ga = mlrose_hiive.GARunner(problem=problem,
                     experiment_name="GA",
                     output_directory="../flipflop_problem",
                     seed=None,
                     iteration_list=itersList,
                     max_attempts=1000,
                     population_sizes=[200],
                     mutation_rates=[0.3])
ga_run_stats, ga_run_curves = ga.run()

ga = mlrose_hiive.GARunner(problem=problem,
                     experiment_name="GA_",
                     output_directory="../flipflop_problem",
                     seed=None,
                     iteration_list=itersList,
                     max_attempts=1000,
                     population_sizes=[200],
                     mutation_rates=[0.5])
ga_run_stats1, ga_run_curves1 = ga.run()

ga = mlrose_hiive.GARunner(problem=problem,
                     experiment_name="GA",
                     output_directory="../flipflop_problem",
                     seed=None,
                     iteration_list=itersList,
                     max_attempts=1000,
                     population_sizes=[200],
                     mutation_rates=[0.6])
ga_run_stats2, ga_run_curves2 = ga.run()


plt.figure()
plt.title('GA Runner for flipflop')
plt.plot(ga_run_curves.Fitness, label='Fitness score Mutation = 0.3',color="navy")
plt.plot(ga_run_curves1.Fitness, label='Fitness score Mutation = 0.5',color="red")
plt.plot(ga_run_curves2.Fitness, label='Fitness score Mutation = 0.6',color="green")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'ga_fitness_flipflop.png')


mimic = mlrose_hiive.MIMICRunner(problem=problem,
                           experiment_name="MIMIC",
                           output_directory="../flipflop_problem",
                           seed=None,
                           iteration_list=itersList,
                           population_sizes=[200],
                           max_attempts=500,
                           keep_percent_list=[0.2],
                           use_fast_mimic=True)
mimic_run_stats, mimic_run_curves = mimic.run()

mimic = mlrose_hiive.MIMICRunner(problem=problem,
                           experiment_name="MIMIC",
                           output_directory="../flipflop_problem",
                           seed=None,
                           iteration_list=itersList,
                           population_sizes=[200],
                           max_attempts=500,
                           keep_percent_list=[0.4],
                           use_fast_mimic=True)
mimic_run_stats1, mimic_run_curves1 = mimic.run()

mimic = mlrose_hiive.MIMICRunner(problem=problem,
                           experiment_name="MIMIC",
                           output_directory="../flipflop_problem",
                           seed=None,
                           iteration_list=itersList,
                           population_sizes=[200],
                           max_attempts=500,
                           keep_percent_list=[0.6],
                           use_fast_mimic=True)
mimic_run_stats2, mimic_run_curves2 = mimic.run()



plt.figure()
plt.title('MIMIC Runner for flipflop')
plt.plot(mimic_run_curves.Fitness, label='Fitness score keep_percent = 0.2',color="navy")
plt.plot(mimic_run_curves1.Fitness, label='Fitness score keep_percent = 0.4',color="red")
plt.plot(mimic_run_curves2.Fitness, label='Fitness score keep_percent = 0.6',color="red")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'mimic_fitness_flipflop.png')


a = sa_run_curves.Fitness.max()
b = sa_run_curves1.Fitness.max()
c = sa_run_curves2.Fitness.max()

maxvalue = maximum(a, b, c)

if maxvalue == a:
    best_sa_run_curves = sa_run_curves
elif maxvalue == b:
    best_sa_run_curves = sa_run_curves1
elif maxvalue == c:
    best_sa_run_curves = sa_run_curves2

a = ga_run_curves.Fitness.max()
b = ga_run_curves1.Fitness.max()
c = ga_run_curves2.Fitness.max()

maxvalue = maximum(a, b, c)

if maxvalue == a:
    best_ga_run_curves = ga_run_curves
elif maxvalue == b:
    best_ga_run_curves = ga_run_curves1
elif maxvalue == c:
    best_ga_run_curves = ga_run_curves2

a = mimic_run_curves.Fitness.max()
b = mimic_run_curves1.Fitness.max()
c = mimic_run_curves2.Fitness.max()

maxvalue = maximum(a, b, c)

if maxvalue == a:
    best_mimic_run_curves = mimic_run_curves
elif maxvalue == b:
    best_mimic_run_curves = mimic_run_curves1
elif maxvalue == c:
    best_mimic_run_curves = mimic_run_curves2

print(rhc_run_curves.Fitness.max())
print(best_sa_run_curves.Fitness.max())
print(best_ga_run_curves.Fitness.max())
print(best_mimic_run_curves.Fitness.max())




plt.figure()
plt.title('Comparison for flipflop')
plt.plot(rhc_run_curves.Fitness, label='RHC',color="navy")
plt.plot(best_sa_run_curves.Fitness, label='SA',color="red")
plt.plot(best_ga_run_curves.Fitness, label='GA',color="blue")
plt.plot(best_mimic_run_curves.Fitness, label='MIMIC',color="green")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'fitness_flipflop.png')



plt.figure()
plt.title('Comparison for flipflop Time')
plt.plot(rhc_run_curves.Time, label='RHC',color="navy")
plt.plot(best_sa_run_curves.Time, label='SA',color="red")
plt.plot(best_ga_run_curves.Time, label='GA',color="blue")
plt.plot(best_mimic_run_curves.Time, label='MIMIC',color="green")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'fitness_flipflop_time.png')


print(rhc_run_curves.Time.max())
print(best_sa_run_curves.Time.max())
print(best_ga_run_curves.Time.max())
print(best_mimic_run_curves.Time.max())

