
#ifndef MLPP_AUTO_ENCODER_H
#define MLPP_AUTO_ENCODER_H

//
//  AutoEncoder.hpp
//
//  Created by Marc Melikyan on 11/4/20.
//

#include "core/math/math_defs.h"

#include <string>
#include <tuple>
#include <vector>

class MLPPAutoEncoder {
public:
	std::vector<std::vector<real_t>> modelSetTest(std::vector<std::vector<real_t>> X);
	std::vector<real_t> modelTest(std::vector<real_t> x);

	void gradientDescent(real_t learning_rate, int max_epoch, bool UI = false);
	void SGD(real_t learning_rate, int max_epoch, bool UI = false);
	void MBGD(real_t learning_rate, int max_epoch, int mini_batch_size, bool UI = false);

	real_t score();

	void save(std::string fileName);

	MLPPAutoEncoder(std::vector<std::vector<real_t>> inputSet, int n_hidden);

private:
	real_t Cost(std::vector<std::vector<real_t>> y_hat, std::vector<std::vector<real_t>> y);

	std::vector<std::vector<real_t>> Evaluate(std::vector<std::vector<real_t>> X);
	std::tuple<std::vector<std::vector<real_t>>, std::vector<std::vector<real_t>>> propagate(std::vector<std::vector<real_t>> X);
	std::vector<real_t> Evaluate(std::vector<real_t> x);
	std::tuple<std::vector<real_t>, std::vector<real_t>> propagate(std::vector<real_t> x);
	void forwardPass();

	std::vector<std::vector<real_t>> inputSet;
	std::vector<std::vector<real_t>> y_hat;

	std::vector<std::vector<real_t>> weights1;
	std::vector<std::vector<real_t>> weights2;

	std::vector<real_t> bias1;
	std::vector<real_t> bias2;

	std::vector<std::vector<real_t>> z2;
	std::vector<std::vector<real_t>> a2;

	int n;
	int k;
	int n_hidden;
};

#endif /* AutoEncoder_hpp */
