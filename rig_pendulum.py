# %%
from neato.neato import NeatO
from neato.species import Species
import pickle
# %%
brain = NeatO.load('pendulum')
with open('pendulum_best_individual_gen88', 'rb') as f:
        genome = pickle.load(f)
# %%
brain._species = [
    Species(
        brain._hyperparams.max_fitness_history, genome
    )
]
for s in brain._species:
    while len(s._members) < 200:
        m = genome.clone()
        m.mutate(brain._hyperparams.mutation_probabilities)
        s._members.append(m)
# %%
for s in brain._species:
    print(len(s._members))

# %%
brain.save('pendulum_rigged')
# %%
