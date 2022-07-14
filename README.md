# reinforcement-frog
Q-Learning based 🐸 for eating 🐌 in a NxM board.

## Motivation
The primary goal was to understand how Reinforcement Learning (RL) works, and an attempt to implement it without using third-party libraries for the agent or the environment. 

The project was based on:
- [Reinforcement Learning: An Introduction (2018), by R. Sutton and A. Barto](http://incompleteideas.net/book/the-book-2nd.html)
- [Q-Learning](https://en.wikipedia.org/wiki/Q-learning)
- [Capítulo 68 - Algoritmo de Agente Baseado em IA com Reinforcement Learning – Q-Learning - Deep Learning Book](https://www.deeplearningbook.com.br/algoritmo-de-agente-baseado-em-ia-com-reinforcement-learning-q-learning/)
- [Simple Reinforcement Learning: Q-learning](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56)

## Dependencies
- [numpy](https://github.com/numpy/numpy)
- [matplotlib](https://github.com/matplotlib/matplotlib)

## Demo:
![Demonstration](./demo.gif)

## How to:
### Training the agent
```sh
$ chmod +x ./main.py
$ ./main.py
```

### Testing the agent
```sh
$ chmod +x ./test.py
$ ./test.py
```

### Settings
The hyperparameters of the board and the agent are available in ```sh settings.py```
