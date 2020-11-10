# Microsoft MLADS Fall 2020
# Reinforcement Learning with Azure ML: Beyond Games

This repository contains the files and code for the Microsoft MLADS Fall 2020 lab.

## Lab Goals

- Gain a basic understanding of the Reinforcement Learning (RL) functionality and APIs available in the Azure ML Platform.
- Understand a typical use case for optimization and how it is framed as a RL problem.
- Learn how to use the RLLib APIs available in Azure ML to integrate a custom RL environment, a custom policy model, and train the RL agent in a scalable fashion.

## Target Audience

### You will benefit most from this lab if you:

- Have a good understanding about the Reinforcement Learning paradigm in Machine Learning, including the main approaches and techniques.
- Have a good understanding about basic neural network architectures and basic concepts of deep learning.
- Are comfortable with reading and writing Python code.
- Know the basic concepts of Azure ML, and how to setup a workspace and compute instance.

## Lab Requirements

### To reproduce this lab you will need:

1. Access to a [Microsoft Azure](https://azure.microsoft.com/en-us/overview/) subscription.
2. Create an Azure ML Workspace and a Compute Instance to run the code for this lab. You can follow the instructions here: [Create and manage Azure Machine Learning workspaces](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=azure-portal) and [Create and manage an Azure Machine Learning compute instance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-manage-compute-instance?tabs=azure-studio#create). A VM with 2 or 4 CPU cores should be enough for your compute instance.
3. Access the [Azure ML studio UI](https://ml.azure.com/) in your workspace and select *Notebooks*. Then click on *Terminal (preview)*. This will open a terminal window in your running compute instance. From there you can clone this GitHub repository into your workspace.

To gain a better understanding of the RL capabilities on Azure ML, please see [this documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-reinforcement-learning).

You may also want to refer to the [Ray](https://docs.ray.io/en/latest/ray-overview/index.html) and especially the [RLLib](https://docs.ray.io/en/latest/rllib.html) documentations, as they form the basis of the RL functionality provided by Azure ML.

## Use Case Overview

We are going to train RL agents that learn to optimize resource allocation for tasks execution.

### The lab is based on the following paper and code:

- Paper: [Resource Management with Deep Reinforcement Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/01/deeprm_hotnets16.pdf)
- Code: [DeepRM GitHub repository](https://github.com/hongzimao/deeprm) \*

\* We are using only the code needed to reproduce the RL environment:[ environment.py]( https://github.com/hongzimao/deeprm/blob/master/environment.py), [job_distribution.py](https://github.com/hongzimao/deeprm/blob/master/job_distribution.py), and [parameters.py](https://github.com/hongzimao/deeprm/blob/master/parameters.py) after minor modifications as explained in the lab notebooks.

## Lab Overview

### The lab is divided in 4 parts and should be executed in the following order:

**1. Azure ML Setup:** this will configure the necessary networking objects and permissions in your Azure ML workspace, to allow the proper instantiation of the RL infrastructure on Azure ML to perform the agent training.

**2. Environment Exploration:** here we explore the environment by acting and observing the corresponding states. We also analyze how a random policy and a heuristics-based policy behave, to have baselines to compare with trained agents. You can also check the environment rollouts for those environments in the videos provided in the [environment_rollouts](https://github.com/vilcek/MLADS_Fall_2020_RL_on_AML_Lab/blob/main/environment_exploration/environment_rollouts) subfolder.

**3. Agent Training:** here we train RL agents on Azure ML, using one of the available RL algorithms from RLLib and showing how to integrate the training with a custom environment and a custom model for the policy. We also show how to scale out the training process, using multiple worker processes for the environment rollout simulations.

**4: Agent Testing:** here we test the trained agents, analyzing the corresponding learned policies by observing the state transitions in the environment during a rollout.
