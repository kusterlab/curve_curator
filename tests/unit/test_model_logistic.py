import numpy as np
import pandas as pd
import pytest
from curve_curator.models import LogisticModel


class TestBasics:

    def test_setup_blank_model(self):
        LM = LogisticModel()

    def test_setup_preinitialized_models_individual_pec50(self):
        LM = LogisticModel(pec50=7.0)
        # fixed params
        expected_out = {'pec50': 7.0}
        assert expected_out == LM.get_fixed_params()
        # free params
        expected_out = {'slope': None, 'front': None, 'back': None}
        assert expected_out == LM.get_free_params()
        # number of free params
        expected_n = 3
        assert expected_n == LM.n_parameter()
        # fitted params
        expected_out = {}
        assert expected_out == LM.get_fitted_params()
        # all params
        expected_out = {'pec50': 7.0, 'slope': np.nan, 'front': np.nan, 'back': np.nan}
        assert expected_out == LM.get_all_parameters()

    def test_setup_preinitialized_models_individual_slope(self):
        LM = LogisticModel(slope=1.0)
        # fixed params
        expected_out = {'slope': 1.0}
        assert expected_out == LM.get_fixed_params()
        # free params
        expected_out = {'pec50': None, 'front': None, 'back': None}
        assert expected_out == LM.get_free_params()
        # number of free params
        expected_n = 3
        assert expected_n == LM.n_parameter()
        # fitted params
        expected_out = {}
        assert expected_out == LM.get_fitted_params()
        # all params
        expected_out = {'pec50': np.nan, 'slope': 1.0, 'front': np.nan, 'back': np.nan}
        assert expected_out == LM.get_all_parameters()

    def test_setup_preinitialized_models_individual_front(self):
        LM = LogisticModel(front=1.0)
        # fixed params
        expected_out = {'front': 1.0}
        assert expected_out == LM.get_fixed_params()
        # free params
        expected_out = {'pec50': None, 'slope': None, 'back': None}
        assert expected_out == LM.get_free_params()
        # number of free params
        expected_n = 3
        assert expected_n == LM.n_parameter()
        # fitted params
        expected_out = {}
        assert expected_out == LM.get_fitted_params()
        # all params
        expected_out = {'pec50': np.nan, 'slope': np.nan, 'front': 1.0, 'back': np.nan}
        assert expected_out == LM.get_all_parameters()

    def test_setup_preinitialized_models_individual_back(self):
        LM = LogisticModel(back=0.0)
        # fixed params
        expected_out = {'back': 0.0}
        assert expected_out == LM.get_fixed_params()
        # free params
        expected_out = {'pec50': None, 'slope': None, 'front': None}
        assert expected_out == LM.get_free_params()
        # number of free params
        expected_n = 3
        assert expected_n == LM.n_parameter()
        # fitted params
        expected_out = {}
        assert expected_out == LM.get_fitted_params()
        # all params
        expected_out = {'pec50': np.nan, 'slope': np.nan, 'front': np.nan, 'back': 0.0}
        assert expected_out == LM.get_all_parameters()

    def test_setup_preinitialized_models_individual_all(self):
        LM = LogisticModel(pec50=7.0, slope=1.0, front=1.0, back=0.0)
        # fixed params
        expected_out = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        assert expected_out == LM.get_fixed_params()
        # free params
        expected_out = {}
        assert expected_out == LM.get_free_params()
        # number of free params
        expected_n = 0
        assert expected_n == LM.n_parameter()
        # fitted params
        expected_out = {}
        assert expected_out == LM.get_fitted_params()
        # all params
        expected_out = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        assert expected_out == LM.get_all_parameters()

    def test_set_and_reset_fitted_values(self):
        LM = LogisticModel()
        # before
        expected_out = {}
        assert expected_out == LM.get_fitted_params()
        # update
        LM.set_fitted_params([7.0, 1.0, 1.0, 0.0])
        expected_out = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        assert expected_out == LM.get_fitted_params()
        assert expected_out == LM.get_all_parameters()
        # reset
        LM.reset()
        expected_out = {}
        assert expected_out == LM.get_fitted_params()


class TestEquality:

    def test_equality_blank(self):
        LM_1 = LogisticModel()
        LM_2 = LogisticModel()
        assert LM_1 == LM_2

    def test_equality_normal(self):
        LM_1 = LogisticModel(pec50=7.0, slope=1.0, front=1.0, back=0.0)
        LM_2 = LogisticModel(pec50=7.0, slope=1.0, front=1.0, back=0.0)
        assert LM_1 == LM_2

    def test_equality_int_vs_float(self):
        LM_1 = LogisticModel(pec50=7, slope=1, front=1, back=0)
        LM_2 = LogisticModel(pec50=7.0, slope=1.0, front=1.0, back=0.0)
        assert LM_1 == LM_2

    def test_inequality_empty_vs_full(self):
        LM_1 = LogisticModel()
        LM_2 = LogisticModel(pec50=7.0, slope=1.0, front=1.0, back=0.0)
        assert LM_1 != LM_2

    def test_inequality_different_values(self):
        LM = LogisticModel(pec50=7.0, slope=1.0, front=1.0, back=0.0)
        LM_pec50 = LogisticModel(pec50=8.0, slope=1.0, front=1.0, back=0.0)
        LM_slope = LogisticModel(pec50=7.0, slope=2.0, front=1.0, back=0.0)
        LM_front = LogisticModel(pec50=7.0, slope=1.0, front=1.1, back=0.0)
        LM_back = LogisticModel(pec50=7.0, slope=1.0, front=1.0, back=1.0)
        assert LM != LM_pec50
        assert LM != LM_slope
        assert LM != LM_front
        assert LM != LM_back

    def test_equality_with_fixed_and_fitted_values(self):
        LM_1 = LogisticModel()
        LM_2 = LogisticModel(pec50=7.0, slope=1.0, front=1.0, back=0.0)
        assert LM_1 != LM_2
        LM_1.set_fitted_params([7.0, 1.0, 1.0, 0.0])
        assert LM_1 == LM_2


class TestCoreFunction:
    x = np.array([-9.0, -8.0, -7.0, -6.0, -5.0])
    params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0.1}

    def test_invalid_input_type(self):
        LM = LogisticModel()
        with pytest.raises(ValueError):
            LM("invalid_input", self.params['pec50'], self.params['slope'], self.params['front'], self.params['back'])
        with pytest.raises(TypeError):
            LM(self.x, "invalid_input", self.params['slope'], self.params['front'], self.params['back'])
            LM(self.x, self.params['pec50'], "invalid_input", self.params['front'], self.params['back'])
            LM(self.x, self.params['pec50'], self.params['slope'], "invalid_input", self.params['back'])
            LM(self.x, self.params['pec50'], self.params['slope'], self.params['front'], "invalid_input")

    def test_valid_input_x_types(self):
        LM = LogisticModel(**self.params)
        y = LM(self.x)
        assert isinstance(y, np.ndarray)
        y = LM(list(self.x))
        assert isinstance(y, np.ndarray)
        y = LM(-8.0)
        assert isinstance(y, float)

    def test_output_shape(self):
        LM = LogisticModel()
        y = LM(self.x, self.params['pec50'], self.params['slope'], self.params['front'], self.params['back'])
        assert y.shape == (5,)

    def test_output_y(self):
        LM = LogisticModel()
        y = LM(self.x, self.params['pec50'], self.params['slope'], self.params['front'], self.params['back'])
        y_expected = np.array([0.991, 0.918, 0.550, 0.182, 0.109])
        np.testing.assert_almost_equal(y, y_expected, decimal=3)

    def test_output_y_logec50(self):
        pec50 = 5.0
        LM = LogisticModel()
        y = LM(self.x, pec50, self.params['slope'], self.params['front'], self.params['back'])
        y_expected = np.array([1.000, 0.999, 0.991, 0.918, 0.550])
        np.testing.assert_almost_equal(y, y_expected, decimal=3)

    def test_output_y_slope(self):
        slope = 0.5
        LM = LogisticModel()
        y = LM(self.x, self.params['pec50'], slope, self.params['front'], self.params['back'])
        y_expected = np.array([0.918, 0.784, 0.550, 0.316, 0.182])
        np.testing.assert_almost_equal(y, y_expected, decimal=3)

    def test_output_y_front(self):
        front = 2.0
        LM = LogisticModel()
        y = LM(self.x, self.params['pec50'], self.params['slope'], front, self.params['back'])
        y_expected = np.array([1.981, 1.827, 1.050, 0.273, 0.119])
        np.testing.assert_almost_equal(y, y_expected, decimal=3)

    def test_output_y_back(self):
        back = 10.0
        LM = LogisticModel()
        y = LM(self.x, self.params['pec50'], self.params['slope'], self.params['front'], back)
        y_expected = np.array([1.089, 1.818, 5.500, 9.182, 9.911])
        np.testing.assert_almost_equal(y, y_expected, decimal=3)

    def test_output_y_nocurve(self):
        front = 1.0
        back = 1.0
        LM = LogisticModel()
        y = LM(self.x, self.params['pec50'], self.params['slope'], front, back)
        y_expected = np.array([1.000, 1.000, 1.000, 1.000, 1.000])
        np.testing.assert_almost_equal(y, y_expected, decimal=3)

    def test_predefined_values_are_used_correctly(self):
        LM0 = LogisticModel()
        LM1 = LogisticModel(**self.params)
        np.testing.assert_almost_equal(LM0(self.x, **self.params), LM1(self.x), decimal=3)

    def test_predict_function(self):
        LM = LogisticModel(**self.params)
        y = LM.predict(self.x)
        y_expected = np.array([0.991, 0.918, 0.550, 0.182, 0.109])
        np.testing.assert_almost_equal(y, y_expected, decimal=3)

    def test_predict_function_error(self):
        LM = LogisticModel()
        with pytest.raises(ValueError):
            y = LM.predict(self.x)
        # TODO: change values and see it is correctly changed


class TestSetBoundaries:
    x = np.array([-9.0, -8.0, -7.0, -6.0, -5.0])
    bounds = {'pec50_delta': 2.0, 'slope_limits': (1e-2, 10.0), 'y_limits': (1e-3, 1e6), 'noise_limits': (0.0, 20.0)}

    def test_correct_parameters(self):
        LM = LogisticModel()
        boundaries = LM.set_boundaries(self.x, **self.bounds)
        boundaries_expected = {'pec50': None, 'slope': None, 'front': None, 'back': None, 'noise': None}
        assert boundaries.keys() == boundaries_expected.keys()

    def test_correct_values(self):
        LM = LogisticModel()
        boundaries = LM.set_boundaries(self.x, **self.bounds)
        boundaries_expected = {'pec50': (3.0, 11.0), 'slope': (1e-2, 10.0), 'front': (1e-3, 1e6), 'back': (1e-3, 1e6), 'noise': (0.0, 20.0)}
        assert boundaries == boundaries_expected

    def test_invalid_input(self):
        LM = LogisticModel()
        with pytest.raises(AssertionError):
            LM.set_boundaries(self.x, self.bounds['pec50_delta'], (-1.0, 10.0), self.bounds['y_limits'], self.bounds['noise_limits'])
            LM.set_boundaries(self.x, self.bounds['pec50_delta'], self.bounds['slope_limits'], (-1.0, 0.0), self.bounds['noise_limits'])
            LM.set_boundaries(self.x, self.bounds['pec50_delta'], self.bounds['slope_limits'], self.bounds['y_limits'], (-1.0, 1.0))


class TestSetGuess:
    x = np.array([-9.0, -8.0, -7.0, -6.0, -5.0])
    bounds = {'pec50_delta': 2.0, 'slope_limits': (1e-2, 10.0), 'y_limits': (1e-3, 1e6), 'noise_limits': (0.0, 20.0)}
    guess = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.1, 'noise': 0.1}

    def test_correct_parameters(self):
        LM = LogisticModel()
        guess = LM.set_initial_guess(**self.guess)
        guess_expected = {'pec50': None, 'slope': None, 'front': None, 'back': None, 'noise': None}
        assert guess.keys() == guess_expected.keys()

    def test_correct_values_nobounds(self):
        LM = LogisticModel()
        guess = LM.set_initial_guess(**self.guess)
        guess_expected = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.1, 'noise': 0.1}
        assert guess == guess_expected

    def test_correct_values_withbounds(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x, **self.bounds)
        guess = LM.set_initial_guess(**self.guess)
        guess_expected = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.1, 'noise': 0.1}
        assert guess == guess_expected

    def test_correct_values_ec50_outofbounds(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x, **self.bounds)
        guess = LM.set_initial_guess(100.0, self.guess['slope'], self.guess['front'], self.guess['back'], self.guess['noise'])
        guess_expected = {'pec50': 11.0, 'slope': 1.0, 'front': 1.0, 'back': 0.1, 'noise': 0.1}
        assert guess == guess_expected

        guess = LM.set_initial_guess(-100.0, self.guess['slope'], self.guess['front'], self.guess['back'], self.guess['noise'])
        guess_expected = {'pec50': 3.0, 'slope': 1.0, 'front': 1.0, 'back': 0.1, 'noise': 0.1}
        assert guess == guess_expected

    def test_correct_values_slope_outofbounds(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x, **self.bounds)
        guess = LM.set_initial_guess(self.guess['pec50'], -1.0, self.guess['front'], self.guess['back'], self.guess['noise'])
        guess_expected = {'pec50': 8.0, 'slope': 0.01, 'front': 1.0, 'back': 0.1, 'noise': 0.1}
        assert guess == guess_expected

        guess = LM.set_initial_guess(self.guess['pec50'], 1000, self.guess['front'], self.guess['back'], self.guess['noise'])
        guess_expected = {'pec50': 8.0, 'slope': 10.0, 'front': 1.0, 'back': 0.1, 'noise': 0.1}
        assert guess == guess_expected

    def test_correct_values_front_outofbounds(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x, **self.bounds)
        guess = LM.set_initial_guess(self.guess['pec50'], self.guess['slope'], -1.0, self.guess['back'], self.guess['noise'])
        guess_expected = {'pec50': 8.0, 'slope': 1.0, 'front': 1e-3, 'back': 0.1, 'noise': 0.1}
        assert guess == guess_expected

        guess = LM.set_initial_guess(self.guess['pec50'], self.guess['slope'], np.inf, self.guess['back'],self.guess['noise'])
        guess_expected = {'pec50': 8.0, 'slope': 1.0, 'front': 1e6, 'back': 0.1, 'noise': 0.1}
        assert guess == guess_expected

    def test_correct_values_back_outofbounds(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x, **self.bounds)
        guess = LM.set_initial_guess(self.guess['pec50'], self.guess['slope'], self.guess['front'], -1.0, self.guess['noise'])
        guess_expected = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 1e-3, 'noise': 0.1}
        assert guess == guess_expected

        guess = LM.set_initial_guess(self.guess['pec50'], self.guess['slope'], self.guess['front'], np.inf, self.guess['noise'])
        guess_expected = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 1e6, 'noise': 0.1}
        assert guess == guess_expected

    def test_correct_values_noise_outofbounds(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x, **self.bounds)
        guess = LM.set_initial_guess(self.guess['pec50'], self.guess['slope'], self.guess['front'], self.guess['back'], -0.1)
        guess_expected = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.1, 'noise': 0.0}
        assert guess == guess_expected

        guess = LM.set_initial_guess(self.guess['pec50'], self.guess['slope'], self.guess['front'], self.guess['back'], np.inf)
        guess_expected = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.1, 'noise': 20.0}
        assert guess == guess_expected


class TestJacobian:
    x = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0])
    params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0}

    def test_shape(self):
        LM = LogisticModel()
        matrix_x = LM.jacobian_matrix(self.x, **self.params)
        assert matrix_x.shape == (len(self.x), 4)


class TestJacobianOLS:
    x = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0])
    params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0}
    param_names = ['pec50', 'slope', 'front', 'back']

    def test_callable(self):
        LM = LogisticModel()
        y = LM(self.x, **self.params)
        jac = LM.build_jacobian_matrix_ols(self.x, y, self.param_names)
        assert callable(jac)

    def test_shape_1(self):
        LM = LogisticModel()
        y = LM(self.x, **self.params)
        jac = LM.build_jacobian_matrix_ols(self.x, y, self.param_names)
        jac_evaluated = jac(self.params.values())
        assert jac_evaluated.shape == (len(self.param_names),)

    def test_shape_2(self):
        LM = LogisticModel()
        y = LM(self.x, **self.params)
        param_names = ['pec50', 'slope']
        jac = LM.build_jacobian_matrix_ols(self.x, y, param_names)
        jac_evaluated = jac(self.params.values())
        assert jac_evaluated.shape == (len(param_names),)

    def test_minimum(self):
        # if we are at the the true solution, all derivatives must be 0.
        LM = LogisticModel()
        y = LM(self.x, **self.params)
        jac = LM.build_jacobian_matrix_ols(self.x, y, self.param_names)
        jac_evaluated = jac(self.params.values())
        jac_evaluated_expected = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_almost_equal(jac_evaluated, jac_evaluated_expected, decimal=6)


class TestAlternativeGuesses:
    x = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0])
    params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0}
    noise = 0.1
    y = LogisticModel.core(x, **params)

    def test_number_of_guesses(self):
        LM = LogisticModel()
        guess_matrix = np.array([g for g in LM.alternative_guesses(self.x, self.y, self.noise)])
        assert guess_matrix.shape == (len(self.x) + 2, 5)

    def test_correct_pec50_spacing(self):
        LM = LogisticModel()
        guess_matrix = np.array([g for g in LM.alternative_guesses(self.x, self.y, self.noise)])
        pec50s_expected = np.array([7.0, 10.5,  9.5,  8.5,  7.5,  6.5,  5.5,  4.5,  3.5])
        np.testing.assert_almost_equal(guess_matrix[:, 0], pec50s_expected, decimal=6)

    def test_correct_slope(self):
        LM = LogisticModel()
        slope_limit = (0, 5)
        guess_matrix = np.array([g for g in LM.alternative_guesses(self.x, self.y, self.noise, slope_limit)])
        ec50s_expected = np.array(9 * [5])
        np.testing.assert_almost_equal(guess_matrix[:, 1], ec50s_expected, decimal=6)

    def test_correct_noise(self):
        LM = LogisticModel()
        guess_matrix = np.array([g for g in LM.alternative_guesses(self.x, self.y, self.noise)])
        ec50s_expected = np.array(9 * [self.noise])
        np.testing.assert_almost_equal(guess_matrix[:, 4], ec50s_expected, decimal=6)


class TestFindBestGuess:
    x = np.array([-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0])
    noise = 0.1
    # TODO: There could be more extensive testing...

    def test_guess_1(self):
        LM = LogisticModel()
        params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0}
        y = LM(self.x, **params)
        guess_expected = 7.5

        guess = LM.find_best_guess_mle(self.x, y, self.noise)['pec50']
        assert guess == guess_expected

        guess = LM.find_best_guess_ols(self.x, y, self.noise)['pec50']
        assert guess == guess_expected

    def test_guess_2(self):
        LM = LogisticModel()
        params = {'pec50': 4.0, 'slope': 1.0, 'front': 1.0, 'back': 0}
        y = LM(self.x, **params)
        guess_expected = 4.5

        guess = LM.find_best_guess_mle(self.x, y, self.noise)['pec50']
        assert guess == guess_expected

        guess = LM.find_best_guess_ols(self.x, y, self.noise)['pec50']
        assert guess == guess_expected

    def test_guess_3(self):
        LM = LogisticModel()
        params = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 1.0}
        y = LM(self.x, **params)
        guess_expected = 3.5

        guess = LM.find_best_guess_mle(self.x, y, self.noise)['pec50']
        assert guess == guess_expected

        guess = LM.find_best_guess_ols(self.x, y, self.noise)['pec50']
        assert guess == guess_expected


class TestFitting:
    x = np.linspace(-10, -5, 20)
    params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0.1}
    noise = 0.1
    y = LogisticModel.core(x, **params)

    def test_exhaustive_mle_fitting(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x)
        LM.extensively_fit_guesses_mle(self.x, self.y, self.noise)
        pd.testing.assert_series_equal(pd.Series(LM.get_fitted_params()), pd.Series(self.params), atol=1e-4)

    def test_exhaustive_ols_fitting(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x)
        LM.extensively_fit_guesses_ols(self.x, self.y, self.noise)
        pd.testing.assert_series_equal(pd.Series(LM.get_fitted_params()), pd.Series(self.params), atol=1e-4)

    def test_efficiently_mle_fitting(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x)
        LM.efficiently_fit_mle(self.x, self.y, self.noise)
        pd.testing.assert_series_equal(pd.Series(LM.get_fitted_params()), pd.Series(self.params), atol=1e-4)

    def test_efficiently_ols_fitting(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x)
        LM.efficiently_fit_ols(self.x, self.y, self.noise)
        pd.testing.assert_series_equal(pd.Series(LM.get_fitted_params()), pd.Series(self.params), atol=1e-4)

    def test_simple_mle_fitting(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x)
        LM.set_initial_guess(10, 1, 1, 1, 0.1)  # MLE minimization algorithm needs some descent. If it starts to good it will not move.
        LM.fit_mle(self.x, self.y)
        fitted_params = LM.get_fitted_params()
        pd.testing.assert_series_equal(pd.Series(fitted_params), pd.Series(self.params), atol=1e-4)

    def test_simple_ols_fitting(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x)
        LM.find_best_guess_ols(self.x, self.y, self.noise)
        LM.fit_ols(self.x, self.y, self.noise)
        fitted_params = LM.get_fitted_params()
        pd.testing.assert_series_equal(pd.Series(fitted_params), pd.Series(self.params), atol=1e-4)

    def test_basinhopping(self):
        LM = LogisticModel()
        LM.set_boundaries(self.x)
        LM.find_best_guess_ols(self.x, self.y, self.noise)
        LM.basinhopping_ols(self.x, self.y, self.noise)
        fitted_params = LM.get_fitted_params()
        pd.testing.assert_series_equal(pd.Series(fitted_params), pd.Series(self.params), atol=1e-4)


class TestDegreesOfFreedom:
    N = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 100])

    def test_4_paramters_optimized(self):
        LM = LogisticModel()
        dof_n, dof_d = list(zip(*[LM.get_dofs(n, optimized=True) for n in self.N]))
        dof_n_expected = np.full_like(self.N, 5)
        dof_d_expected = np.array([
            1.4047619 ,  2.275     ,  3.31100917,  4.24722222,  5.11149773,
            5.94386228,  6.76175869,  7.57249035,  8.37935884,  9.18398966,
            9.98724577, 10.78961538, 11.59138985, 12.39275099, 13.19381669,
            13.99466594, 21.99819513, 29.99910702, 37.99946959, 45.9996492 ,
            53.99975099, 61.99981416, 77.99988521])
        np.testing.assert_almost_equal(dof_n_expected, dof_n, decimal=6)
        np.testing.assert_almost_equal(dof_d_expected, dof_d, decimal=6)

    def test_4_paramters_not_optimized(self):
        LM = LogisticModel()
        dof_n, dof_d = list(zip(*[LM.get_dofs(n, optimized=False) for n in self.N]))
        dof_n_expected = np.full_like(self.N, 4 - 1)
        dof_d_expected = np.array([n - 4 for n in self.N])
        np.testing.assert_almost_equal(dof_n_expected, dof_n, decimal=6)
        np.testing.assert_almost_equal(dof_d_expected, dof_d, decimal=6)


class TestNoise_Estimation:
    x = np.array([-np.inf, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4])
    params = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}

    def test_no_error(self):
        LM = LogisticModel(**self.params)
        y = LM(self.x)
        noise_expected = 0.0
        noise = LM.estimate_noise(self.x, y)
        np.testing.assert_almost_equal(noise, noise_expected)

    def test_with_error_1(self):
        LM = LogisticModel(**self.params)
        y = LM(self.x) + 1.0
        noise_expected = 1.2970766196371821
        noise = LM.estimate_noise(self.x, y)
        np.testing.assert_almost_equal(noise, noise_expected)

    def test_with_error_10(self):
        LM = LogisticModel(**self.params)
        y = LM(self.x) + 10.0
        noise_expected = 12.970766196371821
        noise = LM.estimate_noise(self.x, y)
        np.testing.assert_almost_equal(noise, noise_expected)

    def test_error_with_more_doses(self):
        x = np.array([-np.inf, -12, -11.5, -11, -10.5, -10,  -9.5, -9,  -8.5, -8,  -7.5, -7,  -6.5, -6,  -5.5, -5,  -4.5, -4])
        LM = LogisticModel(**self.params)
        y = LM(x) + 1.0
        noise_expected = 1.205181318024913
        noise = LM.estimate_noise(x, y)
        np.testing.assert_almost_equal(noise, noise_expected)


class TestAreaUnderTheCurve:
    x = np.array([-9.0, -8.0, -7.0, -6.0, -5.0])

    def test_no_regulation_area(self):
        expected_auc = 1.0

        params = {'pec50': 0.0, 'slope': 1.0, 'front': 1.0, 'back': 0.1}
        LM = LogisticModel(**params)
        auc = LM.calculate_auc(self.x)
        np.testing.assert_almost_equal(auc, expected_auc, decimal=6)

        params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 1.0}
        LM = LogisticModel(**params)
        auc = LM.calculate_auc(self.x)
        np.testing.assert_almost_equal(auc, expected_auc, decimal=6)

    def test_regulation_area(self):
        expected_auc = 0.5
        params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        LM = LogisticModel(**params)
        auc = LM.calculate_auc(self.x)
        np.testing.assert_almost_equal(auc, expected_auc, decimal=6)

        expected_auc = 0.75
        params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0.5}
        LM = LogisticModel(**params)
        auc = LM.calculate_auc(self.x)
        np.testing.assert_almost_equal(auc, expected_auc, decimal=6)

        expected_auc = 1.0
        params = {'pec50': 7.0, 'slope': 1.0, 'front': 2.0, 'back': 0.0}
        LM = LogisticModel(**params)
        auc = LM.calculate_auc(self.x)
        np.testing.assert_almost_equal(auc, expected_auc, decimal=6)

    def test_intercept(self):
        params = {'pec50': 7.0, 'slope': 1.0, 'front': 2.0, 'back': 0.0}
        LM = LogisticModel(**params)

        expected_auc = 1.0
        auc = LM.calculate_auc(self.x, intercept=1.0)
        np.testing.assert_almost_equal(auc, expected_auc, decimal=6)

        expected_auc = 0.5
        auc = LM.calculate_auc(self.x, intercept=2.0)
        np.testing.assert_almost_equal(auc, expected_auc, decimal=6)

    def test_non_finite(self):
        params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        x = np.array([-np.inf, -9.0, -8.0, -7.0, -6.0, -5.0, np.nan])
        LM = LogisticModel(**params)
        np.testing.assert_almost_equal(LM.calculate_auc(self.x), LM.calculate_auc(x), decimal=6)

    def test_not_sorted(self):
        params = {'pec50': 7.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        x1 = self.x[::-1]
        LM = LogisticModel(**params)
        np.testing.assert_almost_equal(LM.calculate_auc(self.x), LM.calculate_auc(x1), decimal=6)


class TestR2:
    x = np.array([-np.inf, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4])

    def no_curve_back(self):
        params = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 1.0}
        LM = LogisticModel(**params)
        y = LM(self.x)
        r2 = LM.calculate_r2(self.x, y)
        r2_expected = 0.0
        np.testing.assert_almost_equal(r2, r2_expected)

    def no_curve_pec50(self):
        params = {'pec50': 0.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        LM = LogisticModel(**params)
        y = LM(self.x)
        r2 = LM.calculate_r2(self.x, y)
        r2_expected = 0.0
        np.testing.assert_almost_equal(r2, r2_expected)

    def curve(self):
        params = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        LM = LogisticModel(**params)
        y = LM(self.x)
        r2 = LM.calculate_r2(self.x, y)
        r2_expected = 1.0
        np.testing.assert_almost_equal(r2, r2_expected, decimal=6)

    def test_nan_robustness(self):
        params = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        LM = LogisticModel(**params)
        y = LM(self.x)
        y[-1] = np.nan
        r2_a = LM.calculate_r2(self.x[:-1], y[:-1])
        r2_b = LM.calculate_r2(self.x, y)
        np.testing.assert_almost_equal(r2_a, r2_b)


class TestRMSE:
    x = np.array([-np.inf, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4])

    def test_no_curve_no_error(self):
        params = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 1.0}
        LM = LogisticModel(**params)
        y = LM(self.x)
        rmse_expected = 0.0
        rmse = LM.calculate_rmse(self.x, y)
        np.testing.assert_almost_equal(rmse, rmse_expected)

    def test_curve_no_error(self):
        params = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        LM = LogisticModel(**params)
        y = LM(self.x)
        rmse_expected = 0.0
        rmse = LM.calculate_rmse(self.x, y)
        np.testing.assert_almost_equal(rmse, rmse_expected)

    def test_curve_with_error(self):
        params = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        LM = LogisticModel(**params)
        y = LM(self.x)
        error = 0.25
        rmse = LM.calculate_rmse(self.x, y+error)
        np.testing.assert_almost_equal(rmse, error)

    def test_curve_with_error_nan(self):
        params = {'pec50': 8.0, 'slope': 1.0, 'front': 1.0, 'back': 0.0}
        LM = LogisticModel(**params)
        y = LM(self.x)
        y[-1] = np.nan
        error = 0.25
        rmse = LM.calculate_rmse(self.x, y+error)
        np.testing.assert_almost_equal(rmse, error)
