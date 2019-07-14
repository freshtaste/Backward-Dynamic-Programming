#ifndef BACKWARD_DP_HPP
#define BACKWARD_DP_HPP

#include <vector>
#include <iostream>
#include <map>

using namespace std;

typedef tuple<int, float> S;
typedef float A;
typedef map<S, map<A, map<S, tuple<float, float> >>> SASTff;
typedef map<S, tuple<float, A>> STfA;
typedef tuple<float, float> Tff;
typedef map<S, Tff> STff;

class BackwardDP {
    vector<SASTff> transitions_rewards;
    map<S, float> terminal_opt_val;
    float gamma;
    vector<STfA> vf_and_policy;
    public:
    BackwardDP(vector<SASTff> transitions_rewards,  map<S, float>terminal_opt_val, float gamma);
    vector<STfA> get_vf_and_policy(void);
};

class poisson {
    float lambda;
    public:
    poisson(float lambda);
    float pmf(int k);
    float prob_run_out(int k);
};

#endif /* BACKWARD_DP_HPP */