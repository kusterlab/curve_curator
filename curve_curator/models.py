# models.py
# Logistic & Mean Model classes
#
# Florian P. Bayer - 2024
#


# Imports
import functools
import warnings
import numpy as np
import scipy
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import svd

# Global parameters
NOISE_LIMITS = (1e-5, 20)  # Variance Range of the Normal error
PEC50_DELTA = 2  # Delta +/- pEC50 around the doses to define the boundary
SLOPE_LIMITS = (0.01, 10)  # Slope Range
Y_LIMITS = (1e-4, 1e6)  # Y-axis Range


class _Model:
    """
    Generic Model Prototype

    A model is a callable object with some extra functions to deal with curve fitting and evaluation.
    The core function describes the specific model: Y = f_core(X).
    It is only for internal use to provide functions and attribute structure to the child classes.

    Attributes
    ----------
    params : dict
        fixed parameter(s) to specify values of the core function.
    func : function
        the core function of the model.
    max_iterations : int
        the maximal number of iterations before the parameter estimation is stopped.
    fitted_params : dict
        estimated parameter(s) to some observed x, y data.
    fitted_params_error : dict
        error of estimated parameter(s) to some observed x, y data.
    bounds : dict
        the boundaries of the parameter space.
    guess : dict
        the initial guess before parameter estimation.
    noise : float
        the noise in the data that was used for parameter estimation.
    likelihood : float
        the negative log likelihood of the model given the data used for parameter estimation.
    """
    @staticmethod
    def core(x, *args, **kwargs):
        return x

    @staticmethod
    def primitive_function(x, c=0.0, **kwargs):
        F_x = 0.5 * x**2 + c
        return F_x

    def __init__(self, **kwargs):
        self.params = kwargs
        self.func = core
        self.max_iterations = None
        self.fitted_params = {}
        self.params_error = {}
        self.bounds = {}
        self.guess = {}
        self.noise = np.nan
        self.likelihood = np.nan

    # Make the model callable
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def reset(self):
        """
        Rests all values.
        """
        self.fitted_params = {}
        self.params_error = {}
        self.noise = np.nan
        self.likelihood = np.nan

    def get_fixed_params(self):
        return {p: v for p, v in self.params.items() if v is not None}

    def get_free_params(self):
        return {p: v for p, v in self.params.items() if v is None}

    def get_fitted_params(self):
        return self.fitted_params

    def get_params_error(self):
        return self.params_error

    def get_all_parameters(self, values=None):
        """
        Get all model parameters in key:value format.
        If values are provided, those will be transformed and supplemented with potential fixed params.
        """
        if values is not None:
            values = iter(values)
            return {p: next(values) if value is None else value for p, value in self.params.items()}
        return {p: self.fitted_params.get(p, np.nan) if value is None else value for p, value in self.params.items()}

    def set_fitted_params(self, values):
        """
        Saves the parameter estimates to the modle and returns the object.
        """
        values = iter(values)
        self.fitted_params = {p: next(values) for p, _ in self.params.items() if _ is None}
        return self.fitted_params

    def set_params_error(self, values):
        """
        Saves the parameter error values to the modle and returns the object.
        """
        values = iter(values)
        self.params_error = {p: next(values) for p in self.params.keys()}
        return self.params_error

    def n_parameter(self, fixed=False):
        """
        Returns the number of free or fixed parameters in the model.
        """
        if fixed:
            return len(self.get_fixed_params())
        return len(self.get_free_params())

    def integrate_response(self, interval, params=None):
        """
        Integrates the response area of the Model for the given interval.

        Parameters
        ----------
        interval : array-like
            integration interval (x0, x1). Must be finite values.
        params : dict, optional
            parameters to test against x & y data. Default is none and it will then take the values stored in the model.

        Returns
        -------
        definite integral
        """
        if (len(interval) != 2) or not all(np.isfinite(interval)):
            raise ValueError('Interval must contain one lower and one upper finite boundary value.')
        if params is None:
            params = self.get_all_parameters()
        f0 = self.primitive_function(interval[0], **params)
        f1 = self.primitive_function(interval[1], **params)
        return f1 - f0

    def get_auc(self, x, intercept=1.0, params=None):
        """
        Calculates the AUC for the log-logistic model in the observed drug range.

        Parameters
        ----------
        x : array-like
            integration interval (x0, x1). Must be finite values.
        intercept : float
            the intercept-value for the calculation of the reference area. It should be the expected unregulated response. Default value is 1.0.
        params : dict, optional
            parameters of the model. Default is none and it will then take the values stored in the model.

        Returns
        -------
        area under the curve
        """
        if params is None:
            params = self.get_all_parameters()
        x = x[np.isfinite(x)]
        interval = (np.min(x), np.max(x))
        curve_area = self.integrate_response(interval, params=params)
        reference_area = intercept * (interval[1] - interval[0])
        auc = curve_area / reference_area
        return auc

    def residuals(self, x, y, params=None):
        """
        Calculates the model residuals of the fitted model for particular data x & y.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observes y-values.
        params : dict, optional
            parameters to test against x & y data. Default is none and it will then take the values stored in the model.

        Returns
        -------
        residuals
        """
        if params is None:
            params = self.get_all_parameters()
        return y - self(x, **params)

    def calculate_sum_squared_residuals(self, x, y, params=None):
        """
        Calculates the root summed squared error of the fitted model for particular data x & y.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observes y-values.
        params : dict, optional
            parameters to test against x & y data. Default is none and it will then take the values stored in the model.

        Returns
        -------
        ssr
        """
        if params is None:
            params = self.get_all_parameters()
        return np.sum(self.residuals(x, y, params) ** 2)

    def calculate_rmse(self, x, y, params=None):
        """
        Calculates the root mean squared error of the fitted model for particular data x & y.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observes y-values.
        params : dict, optional
            parameters to test against x & y data. Default is none and it will then take the values stored in the model.

        Returns
        -------
        rmse
        """
        if params is None:
            params = self.get_all_parameters()
        ss_res = self.calculate_sum_squared_residuals(x, y, params)
        return np.sqrt(ss_res / y.size)

    def calculate_r2(self, x, y, params=None):
        """
        Calculates the R2 of the fitted model for particular data x & y.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observes y-values.
        params : dict, optional
            parameters to test against x & y data. Default is none and it will then take the values stored in the model.

        Returns
        -------
        r2
        """
        if params is None:
            params = self.get_all_parameters()
        ss_residual = self.calculate_sum_squared_residuals(x, y, params)
        ss_total = np.mean(y)
        r2 = 1 - ((ss_residual + 1e-10) / (ss_total + 1e-10))
        r2 = r2 if r2 > 0 else 0
        return r2

    def log_likelihood_function(self, x, y):
        """
        Returns the log-likelihood function given data x & y. The returned function can be executed for a given set of parameters.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observes y-values.

        Returns
        -------
        func(... | x, y) -> negative log-likelihood function
        """
        def inner(parameters):
            kwargs = self.get_all_parameters(parameters[:-1])
            y_hat = self(x, **kwargs)
            # Calculate the negative log-likelihood for normal distribution
            neg_likelihood = - np.nansum(stats.norm.logpdf(y, y_hat, parameters[-1]))
            return neg_likelihood
        return inner

    def cost_function_ols(self, x, y, weights=None):
        """
        Returns the ols cost function given data x & y. The returned function can be executed for a given set of parameters.
        Weights can be applied optionally.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observes y-values.
        weights : array-like, optional
            A weight for each (x,y)-pair. A bigger number corresponds to a higher importance. Default is None.

        Returns
        -------
        func(... | x, y, weights)
        """
        def inner(parameters):
            kwargs = self.get_all_parameters(parameters)
            y_hat = self(x, **kwargs)
            weighted_ssr = np.nansum(weights * (y - y_hat) ** 2)
            return weighted_ssr
        if weights is None:
            weights = np.ones_like(x)
        return inner

    def fit_mle(self, x, y):
        """
        Fits the free parameters of the model to the given data using maximum likelihood estimation.
        Minimization is performed using the gradient-free Nelder-Mead algorithm from scipy.
        Fitted parameters are saved to the object.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observes y-values.

        Returns
        -------
        -log_likelihood at the minimum
        """
        llf = self.log_likelihood_function(x, y)
        parameter_names = list(self.get_free_params().keys()) + ['noise']
        guess = np.array([self.guess[pn] for pn in parameter_names])
        bounds = np.array([self.bounds.get(pn, (-np.inf, np.inf)) for pn in parameter_names])
        result = minimize(llf, x0=guess, bounds=bounds, method='Nelder-Mead', options={'maxiter': self.max_iterations})
        self.set_fitted_params(result.x[:-1])
        self.noise = result.x[-1]
        self.likelihood = result.fun
        return result.fun

    def fit_ols(self, x, y, weights=None):
        """
        Fits the free parameters of the model to the given data using ordinary least squares estimation.
        Minimization is performed using the L-BFGS-B algorithm from scipy supplemented with the OLS Jacobian matrix.
        Fitted parameters are saved to the object.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observes y-values.
        weights : array-like, optional
            A weight for each (x,y)-pair. A bigger number corresponds to a higher importance. Default is None.

        Returns
        -------
        OLS cost value at the minimum.
        """
        parameter_names = list(self.get_free_params().keys())
        guess = np.array([self.guess[pn] for pn in parameter_names])
        bounds = np.array([self.bounds.get(pn, (-np.inf, np.inf)) for pn in parameter_names])
        ols = self.cost_function_ols(x, y, weights)
        jac = self.build_jacobian_matrix_ols(x, y, parameter_names)
        result = minimize(ols, x0=guess, bounds=bounds, jac=jac, method='L-BFGS-B', options={'maxiter': self.max_iterations})
        self.set_fitted_params(result.x)
        return result.fun

    def basinhopping_ols(self, x, y, weights=None):
        """
        Fits the free parameters of the model to the given data using ordinary least squares estimation.
        To find the global minimum the basin-hopping global optimization algorithm from scipy is applied.
        All solutions are stored in the result_memory and only the best solution is returned.
        Local minimization is performed using the L-BFGS-B algorithm from scipy supplemented with the OLS Jacobian matrix.

        Parameters
        ----------
        x : array-like
            observed x-values
        y : array-like
            observes y-values
        weights : array-like, optional
            A weight for each (x,y)-pair. A bigger number corresponds to a higher importance. Default is None.

        Returns
        -------
        OLS cost value at the minimum
        """
        def save_result(d):
            def callback(params, fun, accept):
                d[tuple(params)] = fun
            return callback
        parameter_names = list(self.get_free_params().keys())
        guess = np.array([self.guess[pn] for pn in parameter_names])
        bounds = np.array([self.bounds.get(pn, (-np.inf, np.inf)) for pn in parameter_names])
        ols = self.cost_function_ols(x, y, weights)
        jac = self.build_jacobian_matrix_ols(x, y, parameter_names)
        results = {}
        result_memory = save_result(results)
        scipy.optimize.basinhopping(ols, callback=result_memory, niter=1000, x0=guess, minimizer_kwargs={'jac': jac, 'bounds': bounds, 'method': 'L-BFGS-B'})
        best_params = min(results, key=results.get)
        self.set_initial_guess(*best_params, self.noise)
        self.set_fitted_params(best_params)
        return results[best_params]


class MeanModel(_Model):
    """
    Mean Model

    The Mean model is the most simple real-world model and is used as a reference point in comparisons to more complicated models.
    Its core function describes, that there is no relationship between y and x. It always returns the best estimation of y, which is the mean.
    As there is an analytical solution for OLS estimation, we don't need to minimize, thus overwrite's the base fit_ols method.
    Still MLE can be performed if wanted.

    Attributes
    ----------
    params : dict
        fixed parameter(s) to specify values of the core function.
    func : function
        the core function of the model.
    max_iterations : int
        the maximal number of iterations before the parameter estimation is stopped.
    fitted_params : dict
        estimated parameter(s) to some observed x, y data.
    fitted_params_error : dict
        error of estimated parameter(s) to some observed x, y data.
    bounds : dict
        the boundaries of the parameter space.
    guess : dict
        the initial guess before parameter estimation.
    noise : float
        the noise in the data that was used for parameter estimation.
    likelihood : float
        the negative log likelihood of the model given the data used for parameter estimation.
    """
    @staticmethod
    def core(x, intercept):
        """
        Intercept function. This function is executed when the object is called.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        intercept : float
            y-value for all x values.

        Returns
        -------
        y : array-like
            Response as described by the intercept value for each x value.
        """
        return np.full_like(x, intercept)

    @staticmethod
    def primitive_function(x, intercept, c=0.0):
        """
        primitive function of the mean modle.

        Parameters
        ----------
        x : float
            Input drug concentrations in log space.
        intercept : float
            y-value for all x values.
        c : float
            Constant value of the primitive function

        Returns
        -------
        F_x : float
            antiderivative at x
        """
        F_x = intercept * x + c
        return F_x

    @staticmethod
    def jacobian_matrix(x, intercept):
        """
        Partial derivatives of the core function with respect to the intercept.
        It quantifies how much y changes for f(x| params) if we change params at given x, which is constant 1.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        intercept : float
            y-value for all x values.

        Returns
        -------
        Jacobian Matrix : np.array of shape (|x|, 1)
        """
        jac = np.expand_dims(np.full_like(x, 1), axis=1)
        return jac

    # Initialize with optional fixed values
    def __init__(self, intercept=None, max_iterations=1000):
        """
        MeanModel(intercept=None, max_iterations=3000)

        Instantiation of the mean model. The intercept parameter is optional and adding it will fix the intercept.
        Fixed parameters are not part of the optimization procedure. All other attributes are added, modified, or deleted by dedicated methods.

        Parameters
        ----------
        intercept : float, optional
            Fixing the intercept point of the intercept function to the given value. Default None.
        max_iterations : int, optional
            Maximal number of iterations before the MLE fitting procedure is treminated. Default is 3000 iterations.
        """
        self.params = {'intercept': intercept}
        self.func = self.build_model()
        self.max_iterations = max_iterations
        self.fitted_params = {}
        self.params_error = {}
        self.bounds = {}
        self.guess = {}
        self.noise = np.nan
        self.likelihood = np.nan

    def __repr__(self):
        # str representation
        return f'MeanModel({self.get_all_parameters()})'

    def build_model(self):
        # Add the fixed parameters to the core function of the mean model
        return functools.partial(MeanModel.core, **self.get_fixed_params())

    def calculate_parameter_error(self, x, y):
        """
        Calculates the intercept error given x and y. This is the standard deviation.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observed y-values.

        Returns
        -------
        Standard error for the intercept.
        """
        err = np.std(y)
        self.set_params_error(err)
        return err

    def set_boundaries(self, y_limits=Y_LIMITS, noise_limits=NOISE_LIMITS):
        """
        Sets the boundaries of the model. Values outside the boundaries are not considered for fitting.

        Parameters
        ----------
        y_limits : tuple(lower, upper), optional
            The lower and upper plateau limits. Default is global Y_LIMITS. Responses can never be negative.
        noise_limits : tuple(lower, upper), optional
            The lower and upper noise limits. Default is global NOISE_LIMITS. Noise can never be negative.

        Returns
        -------
        self.bounds
        """
        assert (y_limits[0] >= 0) and (y_limits[1] >= 0)
        assert (noise_limits[0] >= 0) and (noise_limits[1] >= 0)
        self.bounds = {'intercept': y_limits, 'noise': noise_limits}
        return self.bounds

    def limit_to_bounds(self, intercept, noise):
        """
        This function takes the intercept and noise parameter and guarantees that the global boundaries are not exceeded.
        If there is no boundary specified, then there will be no clipping.

        Parameters
        ----------
        intercept : float
            The intercept of the mean model.
        noise : float
            The noise (std) of the data.

        Returns
        -------
        intercept, noise
        """
        if 'intercept' in self.bounds:
            intercept = np.clip(intercept, *self.bounds['intercept'])
        if 'noise' in self.bounds:
            noise = np.clip(noise, *self.bounds['noise'])
        return intercept, noise

    def set_initial_guess(self, intercept, noise=0.1):
        """
        Sets the initial guess of the model. Values outside the boundaries are not possible and will beclippedd to the set bounds.

        Parameters
        ----------
        intercept : float
            The intercept of the mean model.
        noise : float
            The noise (std) of the data.

        Returns
        -------
        self.guess
        """
        intercept, noise = self.limit_to_bounds(intercept, noise)
        self.guess = {'intercept': intercept, 'noise': noise}
        return self.guess

    def build_jacobian_matrix_ols(self, x, y, parameter_names, weights=None):
        """
        Builds the Jacobian matrix, which is a function of the specified free curve parameter_names.
        Evaluating the Jacobian matrix for a particular set of parameters yields the change of the ols cost for changing the given parameter.
        Knowing the Jacobian matrix for OLS optimization will speed up the fitting process.
        Partial derivatives of the ols cost function with respect to intercept (f1)
        It can deal with weights and fixed parameters as these must be present for calculating each derivative.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observed y-values.
        parameter_names : list
            list of the free parameters as string names. e.g. ['intercept'].
        weights : array-like, optional
            A weight for each (x,y)-pair. A bigger number corresponds to a higher importance. Default is None.

        Returns
        -------
        jac(params) -> matrix
        """
        # Partial derivatives of OLS to intercept
        def f_1(params):
            (intercept,) = params
            return 2 * np.nansum((intercept - y) * weights)

        # Matrix constructor
        def matrix(params):
            params = self.get_all_parameters(params).values()
            f_x = {'intercept': f_1}
            with np.errstate(all='ignore'):
                return np.array([f_x[p](params) for p in parameter_names])

        if weights is None:
            weights = np.ones_like(x)
        return matrix

    def fit_ols(self, x, y, weights=None):
        """
        Fits the intercept of the model to the given data using ordinary least squares estimation.
        There is an analytical solution for OLS estimation, we don't need to minimize.
        Fitted parameters are saved to the object.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observes y-values.
        weights : array-like, optional
            A weight for each (x,y)-pair. A bigger number corresponds to a higher importance. Default is None.
            CURRENTLY NOT IMPLEMENTED AND WILL NOT DO ANYTHING!
        Returns
        -------
        OLS cost value at the minimum
        """
        # TODO: currently no weights supported. Make a warning..
        if self.params['intercept'] is None:
            self.set_fitted_params((np.mean(y),))
            self.noise = np.std(y)
            ssr = np.var(y) * y.size
            return ssr
        else:
            self.noise = self.calculate_rmse(x, y)
            ssr = self.calculate_sum_squared_residuals(x, y)
            return ssr


class LogisticModel(_Model):
    """
    Logistic Model class

    The logistic model is the key model for the curve curator pipeline. Its core function is a 4 parameter sigmoid function that relates the log
    concentration x to the observed ratios y. The modle is callable and executes the core function. If desired, all parameters can be fixed to some
    prior value. These values must be given when the object is instantiated. The model can adapt to observations using different fitting procedures.
    The jacobian matrix is written in a way that they can handel fixed parameters as well as weights. The fold change of a model is defined as the
    log2 ratio of the estimated curve between the lowest and highest dose.

    Attributes
    ----------
    params : dict
        fixed parameter(s) to specify values of the core function.
    func : function
        the core function of the model.
    max_iterations : int
        the maximal number of iterations before the parameter estimation is stopped.
    fitted_params : dict
        estimated parameter(s) to some observed x, y data. Initially empty but filled after parameter estimation.
    fitted_params_error : dict
        error of estimated parameter(s) to some observed x, y data. Initially empty but filled after parameter estimation.
    bounds : dict
        the boundaries of the parameter space. Initially empty but must be filled before parameter estimation.
    guess : dict
        the initial guess before parameter estimation. Initially empty but must be filled before parameter estimation.
    noise : float
        the noise in the data that was used for parameter estimation.
    likelihood : float
        the negative log likelihood of the model given the data used for parameter estimation.
    """
    @staticmethod
    def core(x, pec50, slope, front, back):
        """
        4-parameter log-logistic function. This function is executed when the object is called.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log space.
        pec50 : float
            Inflection point of the log-logistic function. pEC50 = - log10(EC50).
        slope : float
            Steepness of the transition between front and back plateau.
        front : float
            Front plateau y for x = -inf
        back : float
            Back plateau y for x = inf

        Returns
        -------
        y : array-like
            Response as described by the input parameters for each x value.
        """
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            y = (front - back) / (1 + 10 ** (slope * (x + pec50))) + back
            return y

    @staticmethod
    def primitive_function(x, pec50, slope, front, back, c=0.0):
        """
        primitive function of the 4-parameter log-logistic function.

        Parameters
        ----------
        x : float
            Input drug concentrations in log space.
        pec50 : float
            Inflection point of the log-logistic function. pEC50 = - log10(EC50).
        slope : float
            Steepness of the transition between front and back plateau.
        front : float
            Front plateau y for x = -inf
        back : float
            Back plateau y for x = inf
        c : float
            Constant value of the primitive function

        Returns
        -------
        F_x : float
            antiderivative value at x
        """
        F_x = front * x - (front - back) / slope * np.log10(1 + 10 ** (slope * (x + pec50))) + c
        return F_x

    @staticmethod
    def jacobian_matrix(x, pec50, slope, front, back):
        """
        Partial derivatives of the core function with respect to pec50, slope, front and back.
        It quantifies how much y changes for f(x| params) if we change params at x.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        pec50 : float
            Inflection point of the log-logistic function.
        slope : float
            Steepness of the transition between front and back plateau.
        front : float
            Front plateau y for x = -inf
        back : float
            Back plateau y for x = inf

        Returns
        -------
        Jacobian matrix : np.array of shape (|x|, k) with k params
        """
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            jac = np.array([
                (np.log(10) * (back - front) *    slope    * 10 ** (slope * (x + pec50))) / (1 + 10 ** (slope * (x + pec50))) ** 2,  # pec50
                (np.log(10) * (back - front) * (x + pec50) * 10 ** (slope * (x + pec50))) / (1 + 10 ** (slope * (x + pec50))) ** 2,  # slope
                1 / (1 + 10 ** (slope * (x + pec50))),  # front
                1 - 1 / (1 + 10 ** (slope * (x + pec50))),  # back
            ]).T
            jac[np.isnan(jac)] = 0.
            return jac

    # Initialize with optional fixed values
    def __init__(self, pec50=None, slope=None, front=None, back=None, max_iterations=3000):
        """
        LogisticModel(pec50=None, slope=None, front=None, back=None, max_iterations=3000)

        Instantiation of the log-logistic model. All initial parameters are optional and adding them will fix the parameter.
        Fixed parameters are not part of the optimization procedure. All other attributes are added, modified, or deleted by dedicated methods.

        Parameters
        ----------
        pec50 : numeric, optional
            Fixing the inflection point of the log-logistic function to the given value. Default None.
        slope : numeric, optional
            Fixing the steepness of the transition between front and back plateau to the given value. Default None.
        front : numeric, optional
            Fixing the front plateau to given value. Default None.
        back : numeric, optional
            Fixing the back plateau to given value. Default None.
        max_iterations : int, optional
            Maximal number of iterations before the fitting procedure is terminated. Default is 3000 iterations.
        """
        self.params = {'pec50': pec50, 'slope': slope, 'front': front, 'back': back}
        self.func = self.build_model()
        self.max_iterations = max_iterations
        self.guess = {}
        self.bounds = {}
        self.fitted_params = {}
        self.fitted_params_error = {}
        self.likelihood = np.nan
        self.noise = np.nan

    def __repr__(self):
        # str representation
        return f'LogisticModel({self.get_all_parameters()})'

    def build_model(self):
        # Add the fixed parameters to the core function of the log-logistic model.
        return functools.partial(LogisticModel.core, **self.get_fixed_params())

    def set_boundaries(self, x_values, pec50_delta=PEC50_DELTA, slope_limits=SLOPE_LIMITS, y_limits=Y_LIMITS, noise_limits=NOISE_LIMITS):
        """
        Sets the boundaries of the model. Values outside the boundaries are not considered for fitting.

        Parameters
        ----------
        x_values : array_like
            Array containing the drug concentrations in log10 space.
        pec50_delta : float, optional
            The trailing distance around the given drug concentrations range. Default is global PEC50_DELTA.
        slope_limits : tuple(lower, upper), optional
            The lower and upper slope limits. Default is global SLOPE_LIMITS. Slopes should never be negative.
        y_limits : tuple(lower, upper), optional
            The lower and upper plateau limits. Default is global Y_LIMITS. Responses can never be negative.
        noise_limits : tuple(lower, upper), optional
            The lower and upper noise limits. Default is global NOISE_LIMITS. Noise can never be negative.

        Returns
        -------
        self.bounds
        """
        assert (slope_limits[0] >= 0) and (slope_limits[1] >= 0)
        assert (y_limits[0] >= 0) and (y_limits[1] >= 0)
        assert (noise_limits[0] >= 0) and (noise_limits[1] >= 0)
        self.bounds = {
            'pec50': (min(-x_values) - pec50_delta, max(-x_values) + pec50_delta),
            'slope': slope_limits,
            'front': y_limits,
            'back': y_limits,
            'noise': noise_limits,
        }
        return self.bounds

    def limit_to_bounds(self, pec50, slope, front, back, noise):
        """
        This function takes the curve parameters and guarantees that the global boundaries are not exceeded.
        If there is no boundary specified, then there will be no clipping.

        Parameters
        ----------
        pec50 : float
            Inflection point of the log-logistic function.
        slope : float
            Steepness of the transition between front and back plateau.
        front : float
            Front plateau y for x = -inf.
        back : float
            Back plateau y for x = inf.
        noise : float
            The noise (std) of the curve.

        Returns
        -------
        pec50, slope, front, back, noise
        """
        if 'pec50' in self.bounds:
            pec50 = np.clip(pec50, *self.bounds['pec50'])
        if 'slope' in self.bounds:
            slope = np.clip(slope, *self.bounds['slope'])
        if 'front' in self.bounds:
            front = np.clip(front, *self.bounds['front'])
        if 'back' in self.bounds:
            back = np.clip(back, *self.bounds['back'])
        if 'noise' in self.bounds:
            noise = np.clip(noise, *self.bounds['noise'])
        return pec50, slope, front, back, noise

    def set_initial_guess(self, pec50, slope, front, back, noise):
        """
        Sets the initial guess of the model. Values outside the boundaries are not possible and will be clipped to the set bounds.

        Parameters
        ----------
        pec50 : float
            Inflection point of the log-logistic function.
        slope : float
            Steepness of the transition between front and back plateau.
        front : float
            Front plateau y for x = -inf.
        back : float
            Back plateau y for x = inf.
        noise : float
            The noise (std) of the curve.

        Returns
        -------
        self.guess
        """
        pec50, slope, front, back, noise = self.limit_to_bounds(pec50, slope, front, back, noise)
        self.guess = {'pec50': pec50, 'slope': slope, 'front': front, 'back': back, 'noise': noise}
        return self.guess

    def calculate_fold_change(self, x, to_control=False):
        """
        Calculate the Curve fold change for the interval of x.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        to_control : bool, optional
            Flag if fold change should be calculated against the control (ratio = 1). Default is False which means min to max fold change.

        Returns
        -------
        log2 fold_change
        """
        if to_control:
            return np.log2(self(max(x), **self.fitted_params))
        return np.log2(self(max(x), **self.fitted_params)) - np.log2(self(min(x), **self.fitted_params))

    def build_jacobian_matrix_ols(self, x, y, parameter_names, weights=None):
        """
        Builds the Jacobian matrix, which is a function of the specified free curve parameter_names.
        Evaluating the Jacobian matrix for a particular set of parameters yields the change of the ols cost for changing the given parameter.
        Knowing the Jacobian matrix for OLS optimization will speed up the fitting process.
        Partial derivatives of the ols cost function with respect to pec50 (f1), slope (f2), front (f3), and back (f4).
        It can deal with weights and fixed parameters as these must be present for calculating each derivative.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        y : array-like
            observed y-values.
        parameter_names : list
            list of the free parameters as string names. e.g. ['pec50', 'slope', 'front', 'back'] for 4-parameter fit.
        weights : array-like, optional
            A weight for each (x,y)-pair. A bigger number corresponds to a higher importance. Default is None.

        Returns
        -------
        jac(params) -> matrix
        """
        # Partial derivatives of OLS to pec50
        def f_1(params):
            p, s, f, b = params
            sxp = s * (x + p)
            x_2 = 2 ** (sxp + 1)
            x_5 = 5 ** sxp
            x_10 = 10 ** sxp
            return np.log(10) * s * (b - f) * np.nansum((x_2 * x_5 * (b * x_10 + f - y * (x_10 + 1)) / (x_10 + 1)**3) * weights)

        # Partial derivatives of OLS to slope
        def f_2(params):
            p, s, f, b = params
            sxp = s * (x + p)
            x_2 = 2 ** (sxp + 1)
            x_5 = 5 ** sxp
            x_10 = 10 ** sxp
            return np.log(10) * (b - f) * np.nansum(((x + p) * x_2 * x_5 * (b * x_10 + f - y * (x_10 + 1)) / (x_10 + 1)**3) * weights)

        # Partial derivatives of OLS to front
        def f_3(params):
            p, s, f, b = params
            sxp = s * (x + p)
            x_10 = 10 ** sxp
            return 2 * np.nansum(((b * x_10 + f - y * (x_10 + 1)) / (x_10 + 1)**2) * weights)

        # Partial derivatives of OLS to back
        def f_4(params):
            p, s, f, b = params
            sxp = s * (x + p)
            x_2 = 2 ** (sxp + 1)
            x_5 = 5 ** sxp
            x_10 = 10 ** sxp
            return np.nansum((x_2 * x_5 * (b * x_10 + f - y * (x_10 + 1)) / (x_10 + 1)**2) * weights)

        # Matrix constructor for variable parameters
        def matrix(params):
            params = self.get_all_parameters(params).values()
            f_x = {'pec50': f_1, 'slope': f_2, 'front': f_3, 'back': f_4}
            with np.errstate(all='ignore'):
                return np.array([f_x[p](params) for p in parameter_names])

        if weights is None:
            weights = np.ones_like(x)
        return matrix

    def calculate_parameter_error(self, x, y):
        """
        Calculates the parameter error given x and y. This is equivalent to the scipy implementation using Moore-Penrose approach.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        y : array-like
            observed y-values.

        Returns
        -------
        Standard error for each parameter.
        """
        parameters = self.get_all_parameters()
        # Do Moore-Penrose inverse of jacobian matrix and discarding zero singular values.
        # Similar to scipy optimize.curvefit procedure
        jac = self.jacobian_matrix(x, **parameters)
        _, s, VT = svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        # fractional covariance matrix
        f_cov = np.dot(VT.T / s ** 2, VT)
        # scale f_cov by reduced chi squared residuals to get covariance matrix
        res = y - self(x, **parameters)
        sse = np.sum(res ** 2) / (len(x) - len(parameters))
        cov = f_cov * sse
        # square root of the diagonal elements yields estimate of the standard deviation of the fit parameters
        err = np.sqrt(np.diag(cov))
        self.set_params_error(err)
        return err

    # Iterative alternative start points
    def alternative_guesses(self, x, y, noise, slope_limits=SLOPE_LIMITS):
        """
        This is a generator that yields alternative starting guesses, given the x and y data. The null guess is a straight line.
        The other guesses systematically build curves with varying pEC50s and best guesses of front and back plateau (=mean).
        It is assured that no guess will ever lay out of the specified boundaries.
        pEC50 is defined as the - log10(ec50) and x is already log10 transformed.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        y : array-like
            observes y-values.
        noise : float
            expected noise of the curve.
        slope_limits : tuple(lower, upper), optional
            The lower and upper slope limits. Default is global SLOPE_LIMITS. Slopes should never be negative.

        Yields
        ------
        guess : tuple(pec50, slope, front, back, noise)
        """
        slope_guess = slope_limits[1]
        y_mean = np.mean(y)

        # null guess
        log_ec50_guess = np.median(x)
        front_guess = y_mean
        back_guess = y_mean
        yield self.limit_to_bounds(-log_ec50_guess, slope_guess, front_guess, back_guess, noise)

        # first outside guess
        log_ec50_guess = x[0] - (x[1] - x[0]) / 2
        front_guess = 1
        back_guess = y_mean
        yield self.limit_to_bounds(-log_ec50_guess, slope_guess, front_guess, back_guess, noise)

        # first pice-wise guess
        log_ec50_guess = x[1] - (x[2] - x[1]) / 2
        front_guess = np.mean(y[0])
        back_guess = np.mean(y[1:])
        yield self.limit_to_bounds(-log_ec50_guess, slope_guess, front_guess, back_guess, noise)

        # pice-wise guesses
        for n in range(2, len(x)):
            log_ec50_guess = np.mean([x[n - 1], x[n]])
            front_guess = np.mean(y[:n])
            back_guess = np.mean(y[n:])
            yield self.limit_to_bounds(-log_ec50_guess, slope_guess, front_guess, back_guess, noise)

        # last outside guess
        log_ec50_guess = x[-1] + (x[-1] - x[-2]) / 2
        front_guess = y_mean
        back_guess = 1
        yield self.limit_to_bounds(-log_ec50_guess, slope_guess, front_guess, back_guess, noise)

    def find_best_guess_mle(self, x, y, noise):
        """
        Find the best starting guess for maximum likelihood estimation among the alternative guesses given x and y data and some noise estimate.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        y : array-like
            observes y-values.
        noise : float
            expected noise of the curve.

        Returns
        -------
        best_guess : tuple(ec50, slope, front, back, noise)
        """
        llf = self.log_likelihood_function(x, y)
        best_likelihood, best_guess = np.inf, None
        for guess in self.alternative_guesses(x, y, noise):
            guess_likelihood = llf(guess)
            if guess_likelihood <= best_likelihood:
                best_guess, best_likelihood = guess, guess_likelihood
        return self.set_initial_guess(*best_guess)

    def find_best_guess_ols(self, x, y, noise, weights=None):
        """
        Find the best starting guess for OLS estimation among the alternative guesses given x and y data and some noise estimate.
        Weights can be applied if desired.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        y : array-like
            observes y-values.
        noise : float
            expected noise of the curve.
        weights : array-like, optional
            A weight for each (x,y)-pair. A bigger number corresponds to a higher importance. Default is None.

        Returns
        -------
        best_guess : tuple(ec50, slope, front, back, noise)
        """
        cost = self.cost_function_ols(x, y, weights=weights)
        best_sse, best_guess = np.inf, None
        for guess in self.alternative_guesses(x, y, noise):
            guess_sse = cost(guess[:-1])
            if guess_sse <= best_sse:
                best_guess, best_sse = guess, guess_sse
        return self.set_initial_guess(*best_guess)

    def extensively_fit_guesses_mle(self, x, y, noise, slopes=None):
        """
        Optimize all alternative guesses in combination to different slopes using maximum likelihood estimation and saves the best solution.
        This increases the chances of finding the global minimum and not getting stuck in some local one at the cost of time.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        y : array-like
            observes y-values.
        slopes : list of floats, optional
            The different slope values that should be tested during the extensive fitting procedure.
            By Default None, 3 different slopes are uses ([SLOPE_LIMITS[0], 1.0, SLOPE_LIMITS[1]]).
        noise : float
            expected noise of the curve.
        """
        # Create default slope list
        if slopes is None:
            slopes = [SLOPE_LIMITS[0], 1.0, SLOPE_LIMITS[1]]
        # Go through all guess slope combinations
        best_fit = {'cost': np.inf, 'p_opt': {}, 'g_opt': {}, 'n_opt': noise}
        for guess in self.alternative_guesses(x, y, noise):
            self.set_initial_guess(*guess)
            for slope in slopes:
                self.guess['slope'] = slope
                cost = self.fit_mle(x, y)
                if cost <= best_fit['cost']:
                    best_fit = {'cost': cost, 'p_opt': self.get_fitted_params().values(), 'g_opt': self.guess, 'n_opt': self.noise}
                if self.params['slope'] is not None:
                    break
        # Save the best fit
        self.set_initial_guess(**best_fit['g_opt'])
        self.set_fitted_params(best_fit['p_opt'])
        self.noise = best_fit['n_opt']
        self.likelihood = best_fit['cost']

    def extensively_fit_guesses_ols(self, x, y, noise, slopes=None, weights=None):
        """
        Optimize all alternative guesses in combination to different slopes using OLS estimation and saves the best solution.
        This increases the chances of finding the global minimum and not getting stuck in some local one at the cost of time.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        y : array-like
            observes y-values.
        noise : float
            expected noise of the curve.
        slopes : list of floats, optional
            The different slope values that should be tested during the extensive fitting procedure.
            By Default None, 3 different slopes are uses ([SLOPE_LIMITS[0], 1.0, SLOPE_LIMITS[1]]).
        weights : array-like, optional
            A weight for each (x,y)-pair. A bigger number corresponds to a higher importance. Default is None.
        """
        # Create default slope list
        if slopes is None:
            slopes = [SLOPE_LIMITS[0], 1.0, SLOPE_LIMITS[1]]
        # Go through all guess slope combinations
        best_fit = {'cost': np.inf, 'p_opt': {}, 'g_opt': {}}
        for guess in self.alternative_guesses(x, y, noise):
            self.set_initial_guess(*guess)
            for slope in slopes:
                self.guess['slope'] = slope
                cost = self.fit_ols(x, y, weights=weights)
                if cost < best_fit['cost']:
                    best_fit = {'cost': cost, 'p_opt': self.get_fitted_params().values(), 'g_opt': self.guess}
                if self.params['slope'] is not None:
                    break
        # Save the best fit
        self.set_initial_guess(**best_fit['g_opt'])
        self.set_fitted_params(best_fit['p_opt'])

    def efficiently_fit_mle(self, x, y, noise, slopes=None):
        """
        Fit using ML estimation and saves the best solution. It uses the best alternative guess with three different slopes.
        This is the best compromise between speed and proportion of reaching the best possible fit.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        y : array-like
            observes y-values.
        noise : float
            expected noise of the curve.
        slopes : list of floats, optional
            The different slope values that should be tested during the extensive fitting procedure.
            By Default None, 3 different slopes are uses ([SLOPE_LIMITS[0], 1.0, SLOPE_LIMITS[1]]).
        """
        # Create default slope list
        if slopes is None:
            slopes = [SLOPE_LIMITS[0], 1.0, SLOPE_LIMITS[1]]
        # From best guess start with different slopes to overcome local minima
        best_guess = self.find_best_guess_mle(x, y, noise)
        best_fit = {'cost': np.inf, 'p_opt': {}, 'g_opt': {}, 'n_opt': np.nan}
        for slope in slopes:
            best_guess['slope'] = slope
            self.set_initial_guess(**best_guess)
            cost = self.fit_mle(x, y)
            if cost < best_fit['cost']:
                best_fit = {'cost': cost, 'p_opt': self.get_fitted_params().values(), 'g_opt': best_guess, 'n_opt': self.noise}
            if self.params['slope'] is not None:
                break
        # Save the best fit
        self.set_initial_guess(**best_fit['g_opt'])
        self.set_fitted_params(best_fit['p_opt'])
        self.noise = best_fit['n_opt']
        self.likelihood = best_fit['cost']

    def efficiently_fit_ols(self, x, y, noise, slopes=None, weights=None):
        """
        Fit using OLS estimation and saves the best solution. It uses the best alternative guess with three different slopes.
        This is the best compromise between speed and proportion of reaching the best possible fit.

        Parameters
        ----------
        x : array-like
            Input array with drug concentrations in log10 space.
        y : array-like
            observes y-values.
        noise : float
            expected noise of the curve.
        slopes : list of floats, optional
            The different slope values that should be tested during the extensive fitting procedure.
            By Default None, 3 different slopes are uses ([SLOPE_LIMITS[0], 1.0, SLOPE_LIMITS[1]]).
        weights : array-like, optional
            A weight for each (x,y)-pair. A bigger number corresponds to a higher importance. Default is None.
        """
        # Create default slope list
        if slopes is None:
            slopes = [SLOPE_LIMITS[0], 1.0, SLOPE_LIMITS[1]]
        # From best guess start with different slopes to overcome local minimums
        best_guess = self.find_best_guess_ols(x, y, noise, weights=weights)
        best_fit = {'cost': np.inf, 'p_opt': {}, 'g_opt': {}}
        for slope in slopes:
            best_guess['slope'] = slope
            self.set_initial_guess(**best_guess)
            cost = self.fit_ols(x, y, weights=weights)
            if cost < best_fit['cost']:
                best_fit = {'cost': cost, 'p_opt': self.get_fitted_params().values(), 'g_opt': best_guess}
            if self.params['slope'] is not None:
                break
        # Save the best fit
        self.set_initial_guess(**best_fit['g_opt'])
        self.set_fitted_params(best_fit['p_opt'])

    def estimate_noise(self, x, y, params=None):
        """
        Estimates the unbiased noise in y given the model for particular data x & y input.

        Parameters
        ----------
        x : array-like
            observed x-values.
        y : array-like
            observes y-values.
        params : dict, optional
            parameters to test against x & y data. Default is none and it will then take the values stored in the model.

        Returns
        -------
        rmse
        """
        if params is None:
            params = self.get_all_parameters()
        ss_res = self.calculate_sum_squared_residuals(x, y, params)
        dof = self.get_dofs(n=y.size)[1]
        return np.sqrt(ss_res / dof)

    def get_dofs(self, n, optimized=True):
        """
        Calculates the degrees of freedom of the particular model given the number of data points n.
        The optimized degrees of freedom yield the most calibrated p-values for the 4-p model.
        You can fix parameters, but the H0 distribution will change depending on the specific parameter, its fixed value, the number of data points,
        and the dose resolution. Please do some initial H0 simulations to get correct dfd, dfn, loc, scale parameters for your particular problem.
        These can be overwritten using the fixed option in the toml file.
        Alternatively, one can always fall back to the classical k-1, N-k approach. Which is only true for linear models!

        Future: Make optimized DoFs available for fixed parameters.

        Parameters
        ----------
        n : int
            number of data points
        optimized : bool, optional
            If optimized DoFs should be calculated. Default True.

        Returns
        -------
        dof_n, dof_d
        """
        # slope adjustment effective for small n in range 5-8 data points.
        # With more than 10 data points the adjustment is basically 0 and not relevant.
        def low_n_slope_adjustment(n):
            return 1 / ((n-4)**4 / n + 4)

        # Make sure there is always enough data points for good DoF calculations.
        if n < (self.n_parameter() + 1):
            raise ValueError('The value n needs to be bigger than the number of model parameters.')

        # Optimized degrees of freedom for 4-p model
        if optimized:
            slope = 0.8
            dof_n = 5
            dof_d = (slope - low_n_slope_adjustment(n)) * (n - 2.5)
            return dof_n, dof_d

        # Standard degrees of freedom
        dof_n = self.n_parameter() - 1
        dof_d = n - self.n_parameter()
        return dof_n, dof_d
