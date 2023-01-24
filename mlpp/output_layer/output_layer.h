
#ifndef MLPP_OUTPUT_LAYER_H
#define MLPP_OUTPUT_LAYER_H

//
//  OutputLayer.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "../activation/activation.h"
#include "../cost/cost.h"

#include <map>
#include <string>
#include <vector>


class OutputLayer {
public:
	OutputLayer(int n_hidden, std::string activation, std::string cost, std::vector<std::vector<double>> input, std::string weightInit, std::string reg, double lambda, double alpha);

	int n_hidden;
	std::string activation;
	std::string cost;

	std::vector<std::vector<double>> input;

	std::vector<double> weights;
	double bias;

	std::vector<double> z;
	std::vector<double> a;

	std::map<std::string, std::vector<double> (MLPPActivation::*)(std::vector<double>, bool)> activation_map;
	std::map<std::string, double (MLPPActivation::*)(double, bool)> activationTest_map;
	std::map<std::string, double (Cost::*)(std::vector<double>, std::vector<double>)> cost_map;
	std::map<std::string, std::vector<double> (Cost::*)(std::vector<double>, std::vector<double>)> costDeriv_map;

	double z_test;
	double a_test;

	std::vector<double> delta;

	// Regularization Params
	std::string reg;
	double lambda; /* Regularization Parameter */
	double alpha; /* This is the controlling param for Elastic Net*/

	std::string weightInit;

	void forwardPass();
	void Test(std::vector<double> x);
};


#endif /* OutputLayer_hpp */
