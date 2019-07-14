# Backward-Dynamic-Programming

This is the README file for a python and C++ program that solve the tabular MDP through backward induction. The algorithms are implemented both in python and C++ and include a general design and a vectorized design. Note that in the general design, the data structures of dictionary (python) and map (C++) are heavily used in order to make the algorithm general enough to deal with all sorts of different types of state space and action space. Meanwhile, the vectorized algorithms for a specific MDP problem were developed to significantly improve the speed.

The algorithms are applied to solve the Optimal Dynamic Pricing of Inventories with Stochastic Demand over Finite Horizons. For example, 10 weeks before Christmas, a supermarket decides to adjust price weekly to optimize its revenue. Assume that after Christmas, the Christmas trees are useless and price can only go down. Also, the customers arrive in Poisson distribution with intensity as a function of price.

