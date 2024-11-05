import numpy as np
import pandas as pd
import pytest
from curve_curator.models import MeanModel


class TestBasics:

    def test_setup_blank_model(self):
        M = MeanModel()
        # fixed params
        expected_out = {}
        assert expected_out == M.get_fixed_params()
        # free params
        expected_out = {'intercept': None}
        assert expected_out == M.get_free_params()
        # number of free params
        expected_n = 1
        assert expected_n == M.n_parameter()
        # fitted params
        expected_out = {}
        assert expected_out == M.get_fitted_params()
        # all params
        expected_out = {'intercept': np.nan}
        assert expected_out == M.get_all_parameters()

    def test_setup_preinitialized_model(self):
        M = MeanModel(intercept=1.0)
        # fixed params
        expected_out = {'intercept': 1.0}
        assert expected_out == M.get_fixed_params()
        # free params
        expected_out = {}
        assert expected_out == M.get_free_params()
        # number of free params
        expected_n = 0
        assert expected_n == M.n_parameter()
        # fitted params
        expected_out = {}
        assert expected_out == M.get_fitted_params()
        # all params
        expected_out = {'intercept': 1.0}
        assert expected_out == M.get_all_parameters()

    def test_set_and_reset_fitted_values(self):
        M = MeanModel()
        # before
        expected_out = {}
        assert expected_out == M.get_fitted_params()
        # update
        M.set_fitted_params([1.0])
        expected_out = {'intercept': 1.0}
        assert expected_out == M.get_fitted_params()
        assert expected_out == M.get_all_parameters()
        # reset
        M.reset()
        expected_out = {}
        assert expected_out == M.get_fitted_params()


class TestEquality:

    def test_equality_blank(self):
        M_1 = MeanModel()
        M_2 = MeanModel()
        assert M_1 == M_2

    def test_equality_normal(self):
        M_1 = MeanModel(intercept=1.0)
        M_2 = MeanModel(intercept=1.0)
        assert M_1 == M_2

    def test_equality_int_vs_float(self):
        M_1 = MeanModel(intercept=1.0)
        M_2 = MeanModel(intercept=1)
        assert M_1 == M_2

    def test_inequality_empty_vs_full(self):
        M_1 = MeanModel()
        M_2 = MeanModel(intercept=1.0)
        assert M_1 != M_2

    def test_inequality_different_values(self):
        M_1 = MeanModel(intercept=1.0)
        M_2 = MeanModel(intercept=2.0)
        assert M_1 != M_2

    def test_equality_with_fixed_and_fitted_values(self):
        M_1 = MeanModel()
        M_2 = MeanModel(intercept=1.0)
        assert M_1 != M_2
        M_1.set_fitted_params([1.0])
        assert M_1 == M_2


class TestCoreFunction:
    x = np.array([-9.0, -8.0, -7.0, -6.0, -5.0])
    params = {'intercept': 1}

    def test_invalid_input_type(self):
        M = MeanModel()
        with pytest.raises(TypeError):
            M("invalid_input")

    def test_valid_input_x_types(self):
        M = MeanModel(**self.params)
        y = M(self.x)
        assert isinstance(y, np.ndarray)
        y = M(list(self.x))
        assert isinstance(y, np.ndarray)
        y = M(-8.0)
        assert isinstance(y, type(self.params['intercept']))

    def test_output_shape(self):
        M = MeanModel()
        y = M(self.x, self.params['intercept'])
        assert y.shape == (5,)

    def test_output_y1(self):
        M = MeanModel()
        y = M(self.x, self.params['intercept'])
        y_expected = np.full_like(self.x, 1)
        np.testing.assert_almost_equal(y, y_expected, decimal=3)

    def test_output_y2(self):
        M = MeanModel()
        y = M(self.x, 2)
        y_expected = np.full_like(self.x, 2)
        np.testing.assert_almost_equal(y, y_expected, decimal=3)

    def test_predict_function(self):
        M = MeanModel(**self.params)
        y = M.predict(self.x)
        y_expected = np.full_like(self.x, 1)
        np.testing.assert_almost_equal(y, y_expected, decimal=3)

    def test_predict_function_error(self):
        M = MeanModel()
        with pytest.raises(ValueError):
            y = M.predict(self.x)


class TestInverseFunction:
    y = np.array([0.0, 1.0, 2.0])
    params = {'intercept': 1}

    def test_inverse_function(self):
        M = MeanModel(**self.params)
        x = M.inverse_function(self.y, **M.params)
        x_expected = np.full_like(self.y, np.nan)
        np.testing.assert_almost_equal(x, x_expected, decimal=3)


class TestSetBoundaries:
    x = np.array([-9.0, -8.0, -7.0, -6.0, -5.0])
    bounds = {'y_limits': (1e-3, 1e6), 'noise_limits': (0.0, 20.0)}

    def test_correct_values(self):
        M = MeanModel()
        boundaries = M.set_boundaries(**self.bounds)
        boundaries_expected = {'intercept': (1e-3, 1e6), 'noise': (0.0, 20.0)}
        assert boundaries == boundaries_expected

    def test_invalid_input(self):
        M = MeanModel()
        with pytest.raises(AssertionError):
            M.set_boundaries((-1.0, 0.0), self.bounds['noise_limits'])
            M.set_boundaries(self.bounds['y_limits'], (-1.0, 1.0))

class TestSetGuess:
    x = np.array([-9.0, -8.0, -7.0, -6.0, -5.0])
    bounds = {'y_limits': (1e-3, 1e6), 'noise_limits': (0.0, 20.0)}
    guess = {'intercept': 1.0, 'noise': 0.1}


    def test_correct_parameters(self):
        M = MeanModel()
        guess = M.set_initial_guess(**self.guess)
        guess_expected = {'intercept': None, 'noise': None}
        assert guess.keys() == guess_expected.keys()

    def test_correct_values_nobounds(self):
        M = MeanModel()
        guess = M.set_initial_guess(**self.guess)
        guess_expected = {'intercept': 1.0, 'noise': 0.1}
        assert guess == guess_expected

    def test_correct_values_withbounds(self):
        M = MeanModel()
        M.set_boundaries(**self.bounds)
        guess = M.set_initial_guess(**self.guess)
        guess_expected = {'intercept': 1.0, 'noise': 0.1}
        assert guess == guess_expected

    def test_correct_values_intercept_outofbounds(self):
        M = MeanModel()
        M.set_boundaries(**self.bounds)
        guess = M.set_initial_guess(-1.0, self.guess['noise'])
        guess_expected = {'intercept': 1e-3, 'noise': 0.1}
        assert guess == guess_expected

        guess = M.set_initial_guess(np.inf, self.guess['noise'])
        guess_expected = {'intercept': 1e6, 'noise': 0.1}
        assert guess == guess_expected

    def test_correct_values_noise_outofbounds(self):
        M = MeanModel()
        M.set_boundaries(**self.bounds)
        guess = M.set_initial_guess(self.guess['intercept'], -0.1)
        guess_expected = {'intercept': 1.0, 'noise': 0.0}
        assert guess == guess_expected

        guess = M.set_initial_guess(self.guess['intercept'], np.inf)
        guess_expected = {'intercept': 1.0, 'noise': 20.0}
        assert guess == guess_expected


class TestJacobian:
    x = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0])
    params = {'intercept': 1.0}

    def test_shape(self):
        M = MeanModel()
        matrix_x = M.jacobian_matrix(self.x, **self.params)
        assert matrix_x.shape == (len(self.x), 1)


class TestJacobianOLS:
    x = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0])
    params = {'intercept': 1.0}
    param_names = ['intercept']

    def test_callable(self):
        M = MeanModel()
        y = M(self.x, **self.params)
        jac = M.build_jacobian_matrix_ols(self.x, y, self.param_names)
        assert callable(jac)

    def test_shape_1(self):
        M = MeanModel()
        y = M(self.x, **self.params)
        jac = M.build_jacobian_matrix_ols(self.x, y, self.param_names)
        jac_evaluated = jac(self.params.values())
        assert jac_evaluated.shape == (len(self.param_names),)

    def test_shape_2(self):
        M = MeanModel()
        y = M(self.x, **self.params)
        param_names = []
        jac = M.build_jacobian_matrix_ols(self.x, y, param_names)
        jac_evaluated = jac(self.params.values())
        assert jac_evaluated.shape == (len(param_names),)

    def test_minimum(self):
        # if we are at the the true solution, all derivatives must be 0.
        M = MeanModel()
        y = M(self.x, **self.params)
        jac = M.build_jacobian_matrix_ols(self.x, y, self.param_names)
        jac_evaluated = jac(self.params.values())
        jac_evaluated_expected = np.array([0.0])
        np.testing.assert_almost_equal(jac_evaluated, jac_evaluated_expected, decimal=6)


class TestFitting:
    x = np.linspace(-10, -5, 20)
    noise = 0.1
    params = {'intercept': 1.5}
    guess = {'intercept': 1.0, 'noise': 1.0}
    y = MeanModel.core(x, **params)

    def test_simple_mle_fitting(self):
        M = MeanModel()
        M.set_initial_guess(**self.guess) # MLE minimization algorithm needs some discent. If it starts to good it will not move.
        M.fit_mle(self.x, self.y)
        fitted_params = M.get_fitted_params()
        pd.testing.assert_series_equal(pd.Series(fitted_params), pd.Series(self.params), atol=1e-4)

    def test_simple_ols_fitting(self):
        M = MeanModel()
        M.set_initial_guess(**self.guess)
        M.fit_ols(self.x, self.y, self.noise)
        fitted_params = M.get_fitted_params()
        pd.testing.assert_series_equal(pd.Series(fitted_params), pd.Series(self.params), atol=1e-4)

    def test_basinhopping(self):
        M = MeanModel()
        M.set_initial_guess(**self.guess)
        M.basinhopping_ols(self.x, self.y, self.noise)
        fitted_params = M.get_fitted_params()
        pd.testing.assert_series_equal(pd.Series(fitted_params), pd.Series(self.params), atol=1e-4)


class TestR2:
    x = np.log10([0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]) - 9

    def test_output_1(self):
        M = MeanModel(intercept=1)
        y = M(self.x)
        r2 = M.calculate_r2(self.x, y)
        r2_expected = 0.0
        np.testing.assert_almost_equal(r2, r2_expected)

    def test_output_2(self):
        M = MeanModel(intercept=0)
        y = M(self.x)
        r2 = M.calculate_r2(self.x, y)
        r2_expected = 0.0
        np.testing.assert_almost_equal(r2, r2_expected)

    def test_output_3(self):
        M = MeanModel(intercept=1)
        y = M(self.x)
        y[-1] = -np.inf
        r2 = M.calculate_r2(self.x, y)
        r2_expected = 0.0
        np.testing.assert_almost_equal(r2, r2_expected)

    def test_output_4(self):
        M = MeanModel(intercept=0)
        y = np.array([i%2 for i in range(len(self.x))]) - 0.5
        r2 = M.calculate_r2(self.x, y)
        r2_expected = 0.0
        np.testing.assert_almost_equal(r2, r2_expected)

    def test_nan_robustness(self):
        M = MeanModel(intercept=1)
        y = M(self.x)
        y[-1] = np.nan
        r2_a = M.calculate_r2(self.x[:-1], y[:-1])
        r2_b = M.calculate_r2(self.x, y)
        np.testing.assert_almost_equal(r2_a, r2_b)


class TestRMSE:
    x = np.array([-np.inf, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4])

    def test_no_error(self):
        M = MeanModel(intercept=1)
        y = M(self.x)
        rmse_expected = 0.0
        rmse = M.calculate_rmse(self.x, y)
        np.testing.assert_almost_equal(rmse, rmse_expected)

    def test_with_error(self):
        M = MeanModel(intercept=1)
        y = M(self.x)
        error = 0.5
        rmse = M.calculate_rmse(self.x, y+error)
        np.testing.assert_almost_equal(rmse, error)

    def test_with_error_nan(self):
        M = MeanModel(intercept=1)
        y = M(self.x)
        y[-1] = np.nan
        error = 0.5
        rmse = M.calculate_rmse(self.x, y+error)
        np.testing.assert_almost_equal(rmse, error)
