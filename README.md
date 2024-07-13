# CPS-Project_LunarModule

This repository contains the materials needed to reproduce the project we did for the Cyber Physica Systems exam given by Prof. Laura Nenzi.

Briefly, we are trying to use Deep Q Network (DQN) to train a neural network to control a lunar module during the landing procedure.
Then, we use Signal Temporal Logic (STL) to test whether the designed solution meets a set of given specifications.

See ```report.pdf``` for a more detailed description of the problem and the designed solution.
We also provide in this repo:
- ```DQN.py``` contains a class for training a DQN
- ```Falsification.py``` contains a class for falsifying the model
- ```Test.ipynb``` contains the code for running the experiments, i.e. training and falsifying a model.
- ```Lander_trained_model.h5``` is the model we got from training.
- ```Falsification gif``` is a folder containing videos of the experiments that falsified the specifications, divided by each falsification repetition.
