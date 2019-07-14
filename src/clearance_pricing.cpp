#include <iostream>
#include "backward_dp.hpp"
#include <iomanip>
#include "math.h"
#include <sys/time.h>
#include <armadillo>

using namespace std;
using namespace arma;

BackwardDP get_clearance_backward_dp (int time_steps, int init_inv, vector<Tff> el);
field<mat> get_clearance_backward_dp_vectorized (int time_steps, int init_inv, vector<Tff> el);
// Test the performance
void test_backward_dp (int time_steps, int init_inv, int num_actions);
void test_backward_dp_vectorized (int time_steps, int init_inv, int num_actions);
// Utility functions;
mat get_q_func(int num_states, vector<Tff> el, mat next_vf);
// Helper functions:
float el_func (float x, float alpha, float beta);
vector<Tff> get_el(int n_price);
cube get_tr_prob_helper(int num_states, int num_prices);
cube get_reward_helper(int num_states, int num_prices);
double poisson_pmf(float lambda, int k);
double poisson_prob_run_out(float lambda, int k);

float alpha = 1.0;
float beta = 5.0;

int main(){
    int time_steps = 20;
    int init_inv = 18;
    int n_price=50;
    cout << "Testing the performance of general backward DP and vectorized backwar DP algo. " << endl;
    cout << "Number of time steps: " << time_steps << endl;
    cout << "Initial inventory: " << init_inv << endl;
    cout << "Number of actions: " << n_price << endl;
    cout << "1. Testing general backward DP: "<< endl;
    test_backward_dp (time_steps, init_inv, n_price);
    cout << "2. Testing vectorized backward DP: "<< endl;
    test_backward_dp_vectorized (time_steps, init_inv, n_price);
    return 0;
}


// General Backward DP algorithms
BackwardDP get_clearance_backward_dp (int time_steps, int init_inv, vector<Tff> el) {
    vector<poisson> rvs;
    for (auto item : el) {
        rvs.push_back(poisson (get<1>(item)));
    }
    int num_el = (int)el.size();
    SASTff tr_rew_dict;
    #pragma omp parallel for
    for (int s=0; s < init_inv + 1; s++) {
        for (int p=0; p < num_el; p++) {
            map<A, map<S, Tff>> astff;
            for (int p1=p; p1 < num_el; p1++ ) {
                map<S, Tff> stff;
                for (int d=0; d < s + 1; d++) {
                    if (d < s) {
                        stff[S (s-d, p1)] = Tff (rvs[p1].pmf(d), d * (1 - get<0>(el[p1])));
                    }
                    else {
                        stff[S (s-d, p1)] = Tff (rvs[p1].prob_run_out(d), d * (1 - get<0>(el[p1]))); 
                    }
                }
                astff[p1] = stff;
            }
            tr_rew_dict[S (s, p)] = astff;
        }
    }
    vector<SASTff> transitions_rewards;
    for (int i=0; i< time_steps; i++) {transitions_rewards.push_back(tr_rew_dict);}
    map<S, float> terminal_opt_val;
    for (int s=0; s < init_inv + 1; s++) {
        for (int p=0; p < num_el; p++) {
            terminal_opt_val[S (s, p)] = 0;
        }
    }
    return BackwardDP (transitions_rewards, terminal_opt_val, 1);
}

// Fast, vectorized Backward DP algorithm specifically for this problem
field<mat> get_clearance_backward_dp_vectorized (int time_steps, int init_inv, vector<Tff> el) {
    int num_states = init_inv + 1;
    field<mat> ret(2, time_steps);
    mat vf(init_inv+1, el.size(), fill::zeros);
    for (int t=0; t<time_steps; t++) {
        // initialize policy matrix and value function for the pervious time
        mat pol(init_inv+1, el.size(), fill::zeros);
        mat vf_new = vf;
        mat q_func = get_q_func(num_states, el, vf);
        #pragma omp parallel for
        for (int i=0; i<el.size(); i++) {
            vf_new.col(i) = max(q_func.cols(0, i), 1);
            pol.col(i) = conv_to<vec>::from( index_max(q_func.cols(0, i), 1) );
        }
        vf = vf_new;
        ret(0, time_steps-1-t) = vf;
        ret(1, time_steps-1-t) = pol;
    }
    return ret;
}

// Test the performance
void test_backward_dp (int time_steps, int init_inv, int num_prices) {
    timeval t0,t1;
    vector<Tff> el = get_el(num_prices);
    gettimeofday(&t0, 0);
    vector<STfA> vf_and_pol2 = get_clearance_backward_dp(time_steps, init_inv, el).get_vf_and_policy();
    S init_state (init_inv, 0);
    cout << "Optimal value function at initial state: " << get<0> (vf_and_pol2[0][init_state]) <<endl;
    cout << "Optimal policy at initial state: "<< 1-get<0>(el[(int)get<1>(vf_and_pol2[0][init_state])]) <<endl;
    gettimeofday(&t1, 0);
    double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
    cout << "Time: " <<elapsed << endl;
    
}

void test_backward_dp_vectorized (int time_steps, int init_inv, int num_prices){
    timeval t0,t1;
    vector<Tff> el = get_el(num_prices);
    gettimeofday(&t0, 0);
    field<mat> vf_and_pol = get_clearance_backward_dp_vectorized(time_steps, init_inv, el);
    gettimeofday(&t1, 0);
    double elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
    cout <<  "Optimal value function at initial state: " << vf_and_pol(0,0)(init_inv, num_prices) << endl;
    cout << "Optimal policy at initial state: " << 1-get<0>(el[(int)vf_and_pol(1,0)(init_inv, num_prices)]) << endl;
    cout << "Time: " <<elapsed << endl;
}

// Ultility functions:
mat get_q_func(int num_states, vector<Tff> el, mat next_vf) {
    cube prob = get_tr_prob_helper(num_states, el.size());
    cube reward = get_reward_helper(num_states, el.size());
    cube next_value(num_states, num_states, el.size(), fill::zeros);
    // get next value function
    for (int n=0; n<num_states; n++) {
            next_value.row(n) = next_vf;
    }
    #pragma omp parallel for
    for (int i=0; i<el.size(); i++) {
        float lambda = get<1>(el[i]);
        float price = 1 - get<0>(el[i]);
        // get probability matrix
        vec helper = linspace<vec>(0, num_states-1, num_states);
        prob.slice(i).transform([&, i](double val) { return poisson_pmf(lambda, (int)val); } );
        helper.transform([&, i](double val) { return poisson_prob_run_out(lambda, (int)val+1); } );
        prob.slice(i).col(0) += helper;
        // get reward matrix
        reward.slice(i) *= price;
    }
    mat q_func = sum(prob % (reward + next_value), 1);
    return q_func;
}

// Helper functions: 
float el_func (float x, float alpha, float beta){
    return alpha * pow(M_E, - beta * x);
}

vector<Tff> get_el(int n_price){
    vector<float> price_list;
    for (float i=0; i< n_price+1; i++){
        price_list.push_back(i/n_price);
    }
    reverse(price_list.begin(), price_list.end());
    vector<Tff> el;
    for (auto p: price_list) {
        el.push_back(Tff (1-p, el_func(p, alpha, beta) ) );
    }
    return el;
}

cube get_tr_prob_helper(int num_states, int num_actions) {
    cube ret(num_states, num_states, num_actions, fill::zeros);
    #pragma omp parallel for
    for (int n=0; n<num_actions; n++) {
        for (int i=0; i< num_states; i++){
            for (int j=0; j< num_states; j++) {
                ret(i,j, n) = i-j;
            }
        }
    }
    return ret;
}

cube get_reward_helper(int num_states, int num_actions) {
    cube ret(num_states, num_states, num_actions, fill::zeros);
    #pragma omp parallel for
    for (int n=0; n<num_actions; n++) {
        for (int i=0; i< num_states; i++){
            for (int j=0; j< num_states; j++) {
                ret(i,j,n) =  max(i-j,0);
            }
        }
    }
    return ret;
}

double poisson_pmf(float lambda, int k){
    return (double)pow(M_E, k * log(lambda) - lambda - lgamma(k + 1.0));
}

double poisson_prob_run_out(float lambda, int k){
    // use pmf to approximate probability of running out of inventory
    double ret = 0;
    for (int i=0; i<11; i++) {ret += poisson_pmf(lambda, k+i);}
    return ret;
}