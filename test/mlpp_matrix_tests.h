#ifndef MLPP_MATRIX_TESTS_H
#define MLPP_MATRIX_TESTS_H

// TODO port this class to use the test module once it's working
// Also don't forget to remove it's bindings

#include "core/math/math_defs.h"

#include "core/containers/vector.h"

#include "core/object/reference.h"

#include "core/string/ustring.h"

#include "mlpp_tests.h"

class MLPPMatrix;
class MLPPVector;

class MLPPMatrixTests : public MLPPTests {
	GDCLASS(MLPPMatrixTests, MLPPTests);

public:
	void run_tests();

	void test_mlpp_matrix();

	void test_row_add();
	void test_row_add_pool_vector();
	void test_row_add_mlpp_vector();
	void test_rows_add_mlpp_matrix();

	void test_row_remove();
	void test_row_remove_unordered();


	void test_mlpp_matrix_mul();

	MLPPMatrixTests();
	~MLPPMatrixTests();

protected:
	static void _bind_methods();
};

#endif
