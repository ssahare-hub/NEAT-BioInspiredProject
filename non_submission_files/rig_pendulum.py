# %%
from neato.neato import NeatO
from neato.species import Species
import pickle
# %%
neato = NeatO.load('neato_pendulum')
print(neato._global_best.get_fitness())
# with open('pendulum_best_individual_gen88', 'rb') as f:
#         genome = pickle.load(f)
# # %%
# neato._species = [
#     Species(
#         neato._hyperparams.max_fitness_history, genome
#     )
# ]
# for s in neato._species:
#     while len(s._members) < 200:
#         m = genome.clone()
#         m.mutate(neato._hyperparams.mutation_probabilities)
#         s._members.append(m)
# # %%
# for s in neato._species:
#     print(len(s._members))

# # %%
# neato.save('pendulum_rigged')
# %%
print(neato._global_best.get_fitness())
print(neato._global_best.get_nodes())
print(neato._global_best.get_connections())
# neato._global_best.set_fitness(-2000)
print(neato._global_best.get_fitness())
# %%
neato.save('neato_pendulum')

# %%
