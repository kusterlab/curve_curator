import numpy as np
import pandas as pd
import pytest
from curve_curator.models import MeanModel


class TestCore:
    x = np.array([-9.0, -8.0, -7.0, -6.0, -5.0])
    params = {'intercept': 1}

    def test_invalid_input_type(self):
        M = MeanModel()
        with pytest.raises(TypeError):
            M("invalid_input")

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
