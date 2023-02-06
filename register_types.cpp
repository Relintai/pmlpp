/*
Copyright (c) 2023-present Péter Magyar
Copyright (c) 2022 Marc Melikyan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "register_types.h"

#include "mlpp/data/data.h"
#include "mlpp/lin_alg/mlpp_matrix.h"
#include "mlpp/lin_alg/mlpp_vector.h"

#include "mlpp/activation/activation.h"
#include "mlpp/cost/cost.h"
#include "mlpp/regularization/reg.h"
#include "mlpp/utilities/utilities.h"

#include "mlpp/hidden_layer/hidden_layer.h"
#include "mlpp/multi_output_layer/multi_output_layer.h"
#include "mlpp/output_layer/output_layer.h"

#include "mlpp/kmeans/kmeans.h"
#include "mlpp/knn/knn.h"
#include "mlpp/wgan/wgan.h"

#include "mlpp/mlp/mlp.h"

#include "test/mlpp_tests.h"

void register_pmlpp_types(ModuleRegistrationLevel p_level) {
	if (p_level == MODULE_REGISTRATION_LEVEL_SCENE) {
		ClassDB::register_class<MLPPVector>();
		ClassDB::register_class<MLPPMatrix>();

		ClassDB::register_class<MLPPUtilities>();
		ClassDB::register_class<MLPPReg>();
		ClassDB::register_class<MLPPActivation>();
		ClassDB::register_class<MLPPCost>();

		ClassDB::register_class<MLPPHiddenLayer>();
		ClassDB::register_class<MLPPOutputLayer>();
		ClassDB::register_class<MLPPMultiOutputLayer>();

		ClassDB::register_class<MLPPKNN>();
		ClassDB::register_class<MLPPKMeans>();

		ClassDB::register_class<MLPPMLP>();
		ClassDB::register_class<MLPPWGAN>();

		ClassDB::register_class<MLPPDataESimple>();
		ClassDB::register_class<MLPPDataSimple>();
		ClassDB::register_class<MLPPDataComplex>();
		ClassDB::register_class<MLPPData>();

		ClassDB::register_class<MLPPTests>();
	}
}

void unregister_pmlpp_types(ModuleRegistrationLevel p_level) {
}
