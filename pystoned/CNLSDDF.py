# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint
from pyomo.core.expr.numvalue import NumericValue
import pandas as pd
import numpy as np

from . import CNLS
from .constant import CET_ADDI, FUN_COST, FUN_PROD, RTS_VRS,RTS_CRS, OPT_DEFAULT, OPT_LOCAL
from .utils import tools


class CNLSDDF(CNLS.CNLS):
    """Convex Nonparametric Least Square with directional distance function
    """

    def __init__(self, y, x, b=None, z=None, gy=[1], gx=[1], gb=None, fun=FUN_PROD, rts=RTS_VRS):
        """CNLS DDF model

        Args:
            y (float): output variable.
            x (float): input variables.
            b (float), optional): undesirable output variables. Defaults to None.
            z (float, optional): Contextual variable(s). Defaults to None.
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to None.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.y, self.x, self.b, self.z,self.gy, self.gx, self.gb = \
            tools.assert_valid_direciontal_data_with_z(y,x,b,z,gy,gx,gb)
        self.fun = fun
        self.rts = rts
    
        self.__model__ = ConcreteModel()

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.y)))
        self.__model__.J = Set(initialize=range(len(self.x[0])))
        self.__model__.K = Set(initialize=range(len(self.y[0])))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.beta = Var(
            self.__model__.I, self.__model__.J, bounds=(0.0, None), doc='beta')
        self.__model__.epsilon = Var(self.__model__.I, doc='residuals')
        self.__model__.gamma = Var(
            self.__model__.I, self.__model__.K, bounds=(0.0, None), doc='gamma')

        if type(self.b) != type(None):
            self.__model__.L = Set(initialize=range(len(self.b[0])))
            self.__model__.delta = Var(
                self.__model__.I, self.__model__.L, bounds=(0.0, None), doc='delta')

        if type(self.z) != type(None):
            # Initialize the set of z
            self.__model__.M = Set(initialize=range(len(self.z[0])))
            # Initialize the variables for z variable
            self.__model__.lamda = Var(self.__model__.M, doc='z coefficient')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        self.__model__.translation_rule = Constraint(self.__model__.I,
                                                     rule=self.__translation_property(),
                                                     doc='translation property')
        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='afriat inequality')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0

    def optimize(self, email=OPT_LOCAL, solver=OPT_DEFAULT):
        """Optimize the function by requested method

        Args:
            email (string): The email address for remote optimization. It will optimize locally if OPT_LOCAL is given.
            solver (string): The solver chosen for optimization. It will optimize with default solver if OPT_DEFAULT is given.
        """
        # TODO(error/warning handling): Check problem status after optimization
        self.problem_status, self.optimization_status = tools.optimize_model(
            self.__model__, email, CET_ADDI, solver)

    def __regression_rule(self):
        """Return the proper regression constraint"""
        if self.rts == RTS_VRS:
            if type(self.b) == type(None):
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                            == model.alpha[i] \
                            + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                            - sum(model.lamda[m] * self.z[i][m] for m in model.M) \
                            - model.epsilon[i]

                    return regression_rule
                elif type(self.z) == type(None):
                    def regression_rule(model, i):
                        return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                            == model.alpha[i] \
                            + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                            - model.epsilon[i]

                    return regression_rule
            elif type(self.b) != type(None):
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                            == model.alpha[i] \
                            + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                            + sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                            - sum(model.lamda[m] * self.z[i][m] for m in model.M) \
                            - model.epsilon[i]

                    return regression_rule

                elif type(self.z) == type(None):
                    def regression_rule(model, i):
                        return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                            == model.alpha[i] \
                            + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                            + sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                            - model.epsilon[i]

                    return regression_rule

        elif self.rts == RTS_CRS:
            if type(self.b) == type(None):
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                            == sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                            - sum(model.lamda[m] * self.z[i][m] for m in model.M) \
                            - model.epsilon[i]

                    return regression_rule
                elif type(self.z) == type(None):
                    def regression_rule(model, i):
                        return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                            == sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                            - model.epsilon[i]

                    return regression_rule
            elif type(self.b) != type(None):
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                            == sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                            + sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                            - sum(model.lamda[m] * self.z[i][m] for m in model.M) \
                            - model.epsilon[i]

                    return regression_rule

                elif type(self.z) == type(None):
                    def regression_rule(model, i):
                        return sum(model.gamma[i, k] * self.y[i][k] for k in model.K) \
                            == sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                            + sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                            - model.epsilon[i]

                    return regression_rule

        raise ValueError("Undefined model parameters.")

    def __translation_property(self):
        """Return the proper translation property"""
        if type(self.b) == type(None):
            def translation_rule(model, i):
                return sum(model.beta[i, j] * self.gx[j] for j in model.J) \
                    + sum(model.gamma[i, k] * self.gy[k] for k in model.K) == 1

            return translation_rule

        elif type(self.b) != type(None):
            def translation_rule(model, i):
                return sum(model.beta[i, j] * self.gx[j] for j in model.J) \
                    + sum(model.gamma[i, k] * self.gy[k] for k in model.K) \
                    + sum(model.delta[i, l] * self.gb[l] for l in model.L) == 1

            return translation_rule

    def __afriat_rule(self):
        """Return the proper afriat inequality constraint"""
        if self.fun == FUN_PROD:
            __operator = NumericValue.__le__
        elif self.fun == FUN_COST:
            __operator = NumericValue.__ge__

        if self.rts == RTS_VRS:
            if type(self.b) == type(None):
                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator(model.alpha[i]
                                      + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                                      - sum(model.gamma[i, k] * self.y[i][k] for k in model.K),
                                      model.alpha[h]
                                      + sum(model.beta[h, j] * self.x[i][j] for j in model.J) \
                                      - sum(model.gamma[h, k] * self.y[i][k] for k in model.K))

                return afriat_rule


            elif type(self.b) != type(None):

                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator(model.alpha[i] \
                                      + sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                                      + sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                                      - sum(model.gamma[i, k] * self.y[i][k] for k in model.K),
                                      model.alpha[h]
                                      + sum(model.beta[h, j] * self.x[i][j] for j in model.J) \
                                      + sum(model.delta[h, l] * self.b[i][l] for l in model.L) \
                                      - sum(model.gamma[h, k] * self.y[i][k] for k in model.K))

                return afriat_rule

        elif self.rts == RTS_CRS:
            if type(self.b) == type(None):
                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator( sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                                      - sum(model.gamma[i, k] * self.y[i][k] for k in model.K),
                                       sum(model.beta[h, j] * self.x[i][j] for j in model.J) \
                                      - sum(model.gamma[h, k] * self.y[i][k] for k in model.K))

                return afriat_rule

            elif type(self.b) != type(None):
                def afriat_rule(model, i, h):
                    if i == h:
                        return Constraint.Skip
                    return __operator( sum(model.beta[i, j] * self.x[i][j] for j in model.J) \
                                      + sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                                      - sum(model.gamma[i, k] * self.y[i][k] for k in model.K),
                                        sum(model.beta[h, j] * self.x[i][j] for j in model.J) \
                                      + sum(model.delta[h, l] * self.b[i][l] for l in model.L) \
                                      - sum(model.gamma[h, k] * self.y[i][k] for k in model.K))

                return afriat_rule

        raise ValueError("Undefined model parameters.")

    def display_gamma(self):
        """Display gamma value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.gamma.display()

    def display_delta(self):
        """Display delta value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_undesirable_output(self.b)
        self.__model__.delta.display()

    def get_gamma(self):
        """Return gamma value by array"""
        tools.assert_optimized(self.optimization_status)
        gamma = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.gamma),
                                                           list(self.__model__.gamma[:, :].value))])
        gamma = pd.DataFrame(gamma, columns=['Name', 'Key', 'Value'])
        gamma = gamma.pivot(index='Name', columns='Key', values='Value')
        return gamma.to_numpy()

    def get_delta(self):
        """Return delta value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_undesirable_output(self.b)
        delta = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.delta),
                                                           list(self.__model__.delta[:, :].value))])
        delta = pd.DataFrame(delta, columns=['Name', 'Key', 'Value'])
        delta = delta.pivot(index='Name', columns='Key', values='Value')
        return delta.to_numpy()
