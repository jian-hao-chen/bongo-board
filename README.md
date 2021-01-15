# Bongo Board
This project is the final assignment of Reinforcement Learning course of NTNU.

## How to install
1. Create a python virtual environment in a directory. (**Optional**)  
To execute with global python environment, go to **step 3**.
```bash
$ virtualenv venv
```
> If you don't have virtualenv, you can install it by executing `pip install virtualenv`.

2. Enter the virtual environment after the environment was created successfully.
```bash
# Windows
$ venv\Scripts\activate.bat
# Linux / MacOS
$ source venv/bin/activate
```

3. Install the required packages.
```bash
(venv) $ pip install -r requirements.txt 
```

## How to execute
### Training
```bash
$ python train.py
```
This command will train a model to make the bongo board stay upright as long as possible.  
By default, this program will run 5000 episodes of training and output an image of training results.

#### Usage
```bash
$ python train.py [-h] [--episodes EPISODES] [--gamma GAMMA] [--alpha ALPHA]
                  [--seed SEED] [--log-interval LOG_INTERVAL]
                  [--no-render] [--a2c]
```
* `-h`, `--help` - shows help message and exit
* `--episodes EPISODES` - max episodes (default: 5000)
* `--gamma GAMMA` - discount factor (default: 0.99)
* `--alpha ALPHA` - learning rate (default: 0.001)
* `--seed SEED` - random seed (default: 9527)
* `--log-interval LOG_INTERVAL` - number of episodes for logging interval (default: 100)
* `--no-render` - set to disable render
* `--a2c` - set to use "Actor-Critic" as policy

### Evaluation
```bash
$ python evaluate.py <model_path>
```
This command evaluates the "Bongo Board" with the given model.

#### Usage
```bash
$ python evaluate.py <model_path> [-h] [--a2c]
```
* `-h`, `--help` - shows help message and exit
* `--a2c` - set to load an "Actor-Critic" model

## Reference
* [OpenAI Gym](https://gym.openai.com/)
* [PyTorch](https://pytorch.org)
* [pytorch/examples/reinforcement_learning](https://github.com/pytorch/examples/tree/master/reinforcement_learning)
* REINFORCE algorithm
  > Williams, R. J. (1988). Toward a theory of reinforcement-learning connectionist systems.  
  > Technical Report NU-CCS-88-3, Northeastern University, College of Computer Science.
* Actor-Critic algorithm
  > Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-critic algorithms.  
  > In Advances in neural information processing systems (pp. 1008-1014).