
#ifndef MLPP_UNI_LIN_REG_H
#define MLPP_UNI_LIN_REG_H

//
//  UniLinReg.hpp
//
//  Created by Marc Melikyan on 9/29/20.
//

#include "core/math/math_defs.h"

#include <vector>


class MLPPUniLinReg {
public:
	MLPPUniLinReg(std::vector<real_t> x, std::vector<real_t> y);
	std::vector<real_t> modelSetTest(std::vector<real_t> x);
	real_t modelTest(real_t x);

private:
	std::vector<real_t> inputSet;
	std::vector<real_t> outputSet;

	real_t b0;
	real_t b1;
};


#endif /* UniLinReg_hpp */
