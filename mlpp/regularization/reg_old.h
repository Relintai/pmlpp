

#ifndef MLPP_REG_OLD_H
#define MLPP_REG_OLD_H

//
//  Reg.hpp
//
//  Created by Marc Melikyan on 1/16/21.
//

#include "core/math/math_defs.h"

#include <string>
#include <vector>

class MLPPRegOld {
public:
	real_t regTerm(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg);
	real_t regTerm(std::vector<std::vector<real_t>> weights, real_t lambda, real_t alpha, std::string reg);

	std::vector<real_t> regWeights(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg);
	std::vector<std::vector<real_t>> regWeights(std::vector<std::vector<real_t>> weights, real_t lambda, real_t alpha, std::string reg);

	std::vector<real_t> regDerivTerm(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg);
	std::vector<std::vector<real_t>> regDerivTerm(std::vector<std::vector<real_t>>, real_t lambda, real_t alpha, std::string reg);

private:
	real_t regDerivTerm(std::vector<real_t> weights, real_t lambda, real_t alpha, std::string reg, int j);
	real_t regDerivTerm(std::vector<std::vector<real_t>> weights, real_t lambda, real_t alpha, std::string reg, int i, int j);
};

#endif /* Reg_hpp */
