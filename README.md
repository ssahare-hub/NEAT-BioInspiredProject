# NEAT-BioInspiredProject
An implementation of Neuro-Evolution of Augmenting Topologies algorithm to train population of agents which accomplish a common goal.



Questions:
When choosing random connections in generateNetwork, same connections can be created again and again, So do we need to refine the random creation logic? 

Notes:
1. Keep parents after cross-over along with children or replace the children with the parents (keep only children, drop parents)
2. Mutate child after cross-over
3. Add a different selection operator (proportionate or ranked selection) along each species
4. Change the way fitness is calculated
5. Improve on the way speciation factor δ is calculated
6. Add selection in crossover and/or cloning
7. Offsprings are produced proportional to the best performing candidates/genomes
8. Species -> change crossover
9. Parallel evaluation of genomes - try to parallelize openai gym using the old brain
10. 
