
#ifndef MLPP_GAUSSIAN_NB_H
#define MLPP_GAUSSIAN_NB_H

//
//  GaussianNB.hpp
//
//  Created by Marc Melikyan on 1/17/21.
//

#include <vector>


class GaussianNB {
public:
	GaussianNB(std::vector<std::vector<double>> inputSet, std::vector<double> outputSet, int class_num);
	std::vector<double> modelSetTest(std::vector<std::vector<double>> X);
	double modelTest(std::vector<double> x);
	double score();

private:
	void Evaluate();

	int class_num;

	std::vector<double> priors;
	std::vector<double> mu;
	std::vector<double> sigma;

	std::vector<std::vector<double>> inputSet;
	std::vector<double> outputSet;

	std::vector<double> y_hat;
};

#endif /* GaussianNB_hpp */
