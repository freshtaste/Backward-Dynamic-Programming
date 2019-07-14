# Backward-Dynamic-Programming

This is the README file for a python and C++ program that solve the tabular MDP through backward induction. The algorithms are implemented both in python and C++ and include a general design and a vectorized design. Note that in the general design, the data structures of dictionary (python) and map (C++) are heavily used in order to make the algorithm general enough to deal with all sorts of different types of state space and action space. Meanwhile, the vectorized algorithms for a specific MDP problem were developed to significantly improve the speed.

The algorithms are applied to solve the Optimal Dynamic Pricing of Inventories with Stochastic Demand over Finite Horizons. For example, 10 weeks before Christmas, a supermarket decides to adjust price weekly to optimize its revenue. Assume that after Christmas, the Christmas trees are useless and price can only go down. Also, the customers arrive in Poisson distribution with intensity as a function of price.

The above clearance pricing problem is similar to the order execution problem in finance. In which, a trader needs to execute an order, such as buy or sell 100 share of stocks, within 10 days. He optimally adjusts the amount of stocks to sell or buy everyday. Meanwhile, the market responds with different prices.

Testing Results:
Number of time steps:  20; Initial inventory:  18; Number of actions:  50
1. Python implementation:
⋅⋅*General Backward DP:
⋅⋅⋅Optimal value function at initial state:  1.4714437862760854; 
⋅⋅⋅Optimal policy at initial state:  0.19999999999999996; 
⋅⋅⋅Time:  22.30093216896057 second
⋅⋅*Vectorized Backward DP:
⋅⋅⋅Optimal value function at initial state:  1.4714638725215468; 
⋅⋅⋅Optimal policy at initial state:  0.2; 
⋅⋅⋅Time:  0.13482403755187988

2. C++ implementation:
⋅⋅*General Backward DP:
⋅⋅⋅Optimal value function at initial state: 1.47144; 
⋅⋅⋅Optimal policy at initial state: 0.2; 
⋅⋅⋅Time: 23.6954
⋅⋅*Vectorized Backward DP:
⋅⋅⋅Optimal value function at initial state: 1.47146; 
⋅⋅⋅Optimal policy at initial state: 0.2; 
⋅⋅⋅Time: 0.029083
