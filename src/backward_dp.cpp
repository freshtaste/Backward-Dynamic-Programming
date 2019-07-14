#include <iostream>
#include "backward_dp.hpp"
#include <algorithm>
#include <iomanip>
#include "math.h"

using namespace std;

BackwardDP::BackwardDP (vector<SASTff> transitions_rewards,  map<S, float>terminal_opt_val, float gamma) {
    this->transitions_rewards = transitions_rewards;
    this->terminal_opt_val = terminal_opt_val;
    this->gamma = gamma;
}

vector<STfA> BackwardDP::get_vf_and_policy() {
    STfA vf_pol;
    vector<STfA> ret;
    for (auto item: this->terminal_opt_val) {
        vf_pol[item.first] = tuple<float, A> (item.second, -1);
    }
    int i = 0;
    for (auto tr: this->transitions_rewards) {
        STfA vf_pol_new;
        for (auto sd: tr) {
            auto s = sd.first;
            auto d = sd.second;
            A action_opt;
            float val_opt=-1;
            for (auto ad1 : d) {
                auto a = ad1.first;
                auto d1 = ad1.second;
                float sum = 0;
                for (auto s1pr : d1) {
                    auto s1 = s1pr.first;
                    auto pr = s1pr.second;
                    float p = get<0> (pr);
                    float r = get<1> (pr);
                    sum += p * (r + this->gamma * get<0> (vf_pol.at(s1)));
                }
                if (sum > val_opt) {
                    val_opt = sum;
                    action_opt = a;
                }
            }
            vf_pol_new[s] = tuple<float, A> (val_opt, action_opt);
        }
        ret.push_back(vf_pol_new);
        vf_pol = vf_pol_new;
        i++;
    }
    reverse(ret.begin(), ret.end());
    return ret;
}

poisson::poisson (float lambda) {
    this->lambda = lambda;
}

float poisson::pmf(int k){
    return (float)pow(M_E, k * log(this->lambda) - this->lambda - lgamma(k + 1.0));
}

float poisson::prob_run_out(int k){
    // use pmf to approximate probability of running out of inventory
    float ret = 0;
    for (int i=0; i<5; i++) {ret += pmf(k+i);}
    return ret;
}