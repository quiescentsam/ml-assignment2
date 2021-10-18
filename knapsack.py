import sys
import six
sys.modules['sklearn.externals.six'] = six
import numpy as np
import mlrose_hiive as mlrose
# import mlrose
import matplotlib.pyplot as plt

# fitness = mlrose.Knapsack(weights=[10, 5, 2, 8, 15], values=[1, 2, 3, 4, 5], max_weight_pct=0.4)
# fitness = mlrose.Knapsack(weights=[2,4], values=[1, 2], max_weight_pct=0.4)
# problem = mlrose.DiscreteOpt(length=5, fitness_fn=fitness, maximize=True, max_val=2)
weights = list(np.random.randint(low=1, high=50, size=9))
values = list(np.random.randint(low=1, high=50, size=9))
fitness = mlrose.Knapsack(weights=weights, values=values, max_weight_pct=0.6)
problem = mlrose.DiscreteOpt(length=9, fitness_fn=fitness, maximize=True, max_val=2)
save_path = "knapsack"

iterations = 2 ** np.arange(7)

rhc = mlrose.RHCRunner(problem=problem,
                       experiment_name="RCH_final",
                       output_directory="knapsack_problem",
                       seed=None,
                       iteration_list=iterations,
                       max_attempts=1000,
                       restart_list=[0])
rhc_run_stats, rhc_run_curves = rhc.run()
#
#(rhc_run_curves)
plt.figure()
plt.title('RHC Runner for Knapsack')
plt.plot(rhc_run_curves.Fitness, label='Fitness score',color="navy")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'rhc_fitness.png')

## SA
sa500 = mlrose.SARunner(problem=problem,
                        experiment_name="SA_final",
                        output_directory="knapsack_problem",
                        seed=None,
                        iteration_list=iterations,
                        max_attempts=1000,
                        temperature_list=[250],
                        decay_list=[mlrose.ExpDecay])
sa_run_stats500, sa_run_curves500 = sa500.run()
#
sa100 = mlrose.SARunner(problem=problem,
                        experiment_name="SA_final",
                        output_directory="knapsack_problem",
                        seed=None,
                        iteration_list=iterations,
                        max_attempts=1000,
                        temperature_list=[100],
                        decay_list=[mlrose.ExpDecay])
sa_run_stats100, sa_run_curves100 = sa100.run()
#
sa10 = mlrose.SARunner(problem=problem,
                       experiment_name="SA_final",
                       output_directory="knapsack_problem",
                       seed=None,
                       iteration_list=iterations,
                       max_attempts=1000,
                       temperature_list=[10],
                       decay_list=[mlrose.ExpDecay])
sa_run_stats10, sa_run_curves10 = sa10.run()
#
sa1 = mlrose.SARunner(problem=problem,
                      experiment_name="SA_final",
                      output_directory="knapsack_problem",
                      seed=None,
                      iteration_list=iterations,
                      max_attempts=1000,
                      temperature_list=[1],
                      decay_list=[mlrose.ExpDecay])
sa_run_stats1, sa_run_curves1 = sa1.run()
#
# print(sa_run_curves)
plt.figure()
plt.title('SA Runner for Knapsack')
plt.plot(sa_run_curves1.Fitness, label='Fitness score - temp 1',color="navy")
plt.plot(sa_run_curves10.Fitness, label='Fitness score - temp 10',color="red")
plt.plot(sa_run_curves100.Fitness, label='Fitness score - temp 100',color="green")
plt.plot(sa_run_curves500.Fitness, label='Fitness score - temp 250',color="blue")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'sa_fitness.png')



best_sa_fitness = (max(max(sa_run_curves1.Fitness),max(sa_run_curves10.Fitness),max(sa_run_curves100.Fitness),
                       max(sa_run_curves500.Fitness)))
if best_sa_fitness == max(sa_run_curves1.Fitness):
    best_sa_run_curves = sa_run_curves1
if best_sa_fitness == max(sa_run_curves10.Fitness):
    best_sa_run_curves = sa_run_curves10
if best_sa_fitness == max(sa_run_curves100.Fitness):
    best_sa_run_curves = sa_run_curves100
if best_sa_fitness == max(sa_run_curves500.Fitness):
    best_sa_run_curves = sa_run_curves500

#
#
ga1 = mlrose.GARunner(problem=problem,
                      experiment_name="GA_final",
                      output_directory="knapsack_problem",
                      seed=None,
                      iteration_list=iterations,
                      max_attempts=100,
                      population_sizes=[100],
                      mutation_rates=[0.1])
ga_run_stats1, ga_run_curves1 = ga1.run()

ga3 = mlrose.GARunner(problem=problem,
                      experiment_name="GA_final",
                      output_directory="knapsack_problem",
                      seed=None,
                      iteration_list=iterations,
                      max_attempts=100,
                      population_sizes=[100],
                      mutation_rates=[0.3])
ga_run_stats3, ga_run_curves3 = ga3.run()

ga5 = mlrose.GARunner(problem=problem,
                      experiment_name="GA_final",
                      output_directory="knapsack_problem",
                      seed=None,
                      iteration_list=iterations,
                      max_attempts=100,
                      population_sizes=[100],
                      mutation_rates=[0.5])
ga_run_stats5, ga_run_curves5 = ga5.run()



#
# print(ga_run_stats)
plt.figure()
plt.title('GA Runner for Knapsack')
plt.plot(ga_run_curves1.Fitness, label='Fitness score - mutation rate 0.1',color="navy")
plt.plot(ga_run_curves3.Fitness, label='Fitness score - mutation rate 0.3',color="red")
plt.plot(ga_run_curves5.Fitness, label='Fitness score - mutation rate 0.5',color="green")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'ga_fitness.png')


best_ga_fitness = (max(max(ga_run_curves1.Fitness),max(ga_run_curves3.Fitness),max(ga_run_curves5.Fitness),
                       ))
if best_ga_fitness == max(ga_run_curves1.Fitness):
    best_ga_run_curves = ga_run_curves1
if best_ga_fitness == max(ga_run_curves3.Fitness):
    best_ga_run_curves = ga_run_curves3
if best_ga_fitness == max(ga_run_curves5.Fitness):
    best_ga_run_curves = ga_run_curves5



#
#
mimic2 = mlrose.MIMICRunner(problem=problem,
                            experiment_name="MIMIC_final",
                            output_directory="knapsack_problem",
                            seed=None,
                            iteration_list=iterations,
                            population_sizes=[200],
                            max_attempts=500,
                            keep_percent_list=[0.2],
                            use_fast_mimic=True)
mimic_run_stats2, mimic_run_curves2 = mimic2.run()

mimic4 = mlrose.MIMICRunner(problem=problem,
                            experiment_name="MIMIC_final",
                            output_directory="knapsack_problem",
                            seed=None,
                            iteration_list=iterations,
                            population_sizes=[200],
                            max_attempts=500,
                            keep_percent_list=[0.4],
                            use_fast_mimic=True)
mimic_run_stats4, mimic_run_curves4 = mimic4.run()

mimic6 = mlrose.MIMICRunner(problem=problem,
                            experiment_name="MIMIC_final",
                            output_directory="knapsack_problem",
                            seed=None,
                            iteration_list=iterations,
                            population_sizes=[200],
                            max_attempts=500,
                            keep_percent_list=[0.6],
                            use_fast_mimic=True)
mimic_run_stats6, mimic_run_curves6 = mimic6.run()

mimic8 = mlrose.MIMICRunner(problem=problem,
                            experiment_name="MIMIC_final",
                            output_directory="knapsack_problem",
                            seed=None,
                            iteration_list=iterations,
                            population_sizes=[200],
                            max_attempts=500,
                            keep_percent_list=[0.8],
                            use_fast_mimic=True)
mimic_run_stats8, mimic_run_curves8 = mimic8.run()

plt.figure()
plt.title('Mimic Runner for Knapsack')
plt.plot(mimic_run_curves2.Fitness, label='Fitness score - keep percent 0.2',color="navy")
plt.plot(mimic_run_curves4.Fitness, label='Fitness score - keep percent 0.4',color="red")
plt.plot(mimic_run_curves6.Fitness, label='Fitness score - keep percent 0.6',color="green")
plt.plot(mimic_run_curves8.Fitness, label='Fitness score - keep percent 0.8',color="blue")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'mimic_fitness.png')


best_mimic_fitness = (max(max(mimic_run_curves2.Fitness),max(mimic_run_curves4.Fitness),max(mimic_run_curves6.Fitness),
                          max(mimic_run_curves8.Fitness)))
if best_mimic_fitness == max(mimic_run_curves2.Fitness):
    best_mimic_fitness_curves = mimic_run_curves2
if best_mimic_fitness == max(mimic_run_curves4.Fitness):
    best_mimic_fitness_curves = mimic_run_curves4
if best_mimic_fitness == max(mimic_run_curves6.Fitness):
    best_mimic_fitness_curves = mimic_run_curves6
if best_mimic_fitness == max(mimic_run_curves8.Fitness):
    best_mimic_fitness_curves = mimic_run_curves8



plt.figure()
plt.title('Alogrithm Comparison for Knapsack')
plt.plot(rhc_run_curves.Fitness, label='RHC',color="navy")
plt.plot(best_sa_run_curves.Fitness, label='SA',color="red")
plt.plot(best_ga_run_curves.Fitness, label='GA',color="blue")
plt.plot(best_mimic_fitness_curves.Fitness, label='MIMIC',color="green")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'fitness.png')
plt.figure()
plt.title('Alogrithm Comparison for Knapsack Time')
plt.plot(rhc_run_curves.Time, label='RHC',color="navy")
plt.plot(best_sa_run_curves.Time, label='SA',color="red")
plt.plot(best_ga_run_curves.Time, label='GA',color="blue")
plt.plot(best_mimic_fitness_curves.Time, label='MIMIC',color="green")
plt.xlabel('Iteration')
plt.ylabel("Fitness")
plt.legend(loc="best")
plt.grid()
plt.savefig(save_path + 'fitness_time.png')