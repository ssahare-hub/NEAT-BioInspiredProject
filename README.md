# NEAT-BioInspiredProject
An implementation of Neuro-Evolution of Augmenting Topologies algorithm to train population of agents which accomplish a common goal.

## How to use
- Install the dependencies by runnning 
  - pip install -r ./requirements.txt
- go to (GraphViz)[https://graphviz.org/download/] and install the executable according to the OS you are using
- We have the following three environments available:
    - Cartpole
    - MountainCar
    - Acrobot
- We have the training and test files for each environment for both frameworks, neato and neat-python
- neato files are in the following format:
  - training files -> <environment_name>Parallel.py
  - test files -> <environment_name>Test.py
- neat-python files are in the following format:
  - training files -> <environment_name>NeatPython.py
  - test files -> <environment_name>NeatPythontest.py
- Once you run any of the training files, new folders will be created according to the environment and in order to test them, you will need to add the path of one of the model files to the test file. For example, if you train on the cartpole environment.
  - run the following command -> 
    - python cartpoleParallel.py
  - a new folder cartpole will be created and will contain neato_cartpole_best_individual file
  - Go to cartpoleTest.py and add the path to the file created before within the line that say 
    - open('<filepath_here>', 'rb') as f:
  - run the following command
    - python cartpoleTest.py
- You see the output of the agent running in the environment.