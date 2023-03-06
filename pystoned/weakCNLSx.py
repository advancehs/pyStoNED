# import dependencies
from pyomo.environ import ConcreteModel, Set, Var, Objective, minimize, Constraint, log
from pyomo.core.expr.numvalue import NumericValue
import numpy as np
import pandas as pd

from .constant import CET_ADDI, CET_MULT, FUN_PROD, FUN_COST, OPT_DEFAULT, RTS_CRS, RTS_VRS, OPT_LOCAL
from .utils import tools
from . import weakCNLS


class weakCNLSx(weakCNLS.weakCNLS):
    """Convex Nonparametric Least Square with weak disposability (weakCNLSx)
    """

    def __init__(self, y, x, b, z=None, cet=CET_ADDI, fun=FUN_PROD, rts=RTS_VRS):
        """weakCNLSx model
        lnx=ln(\gamma y -\delta b -\alpha) - \epsilon(\epsilon<0)

        Args:
            y (float): output variable. 
            x (float): input variables.
            b (float): undersiable variables.
            z (float, optional): Contextual variable(s). Defaults to None.
            cet (String, optional): CET_ADDI (additive composite error term) or CET_MULT (multiplicative composite error term). Defaults to CET_ADDI.
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.y, self.x, self.b, self.z = tools.assert_valid_wp_data(y, x, b, z)

        self.cet = cet
        self.fun = fun
        self.rts = rts

        # Initialize the CNLS model
        self.__model__ = ConcreteModel()

        if type(self.z) != type(None):
            # Initialize the set of z
            self.__model__.K = Set(initialize=range(len(self.z[0])))

            # Initialize the variables for z variable
            self.__model__.lamda = Var(self.__model__.K, doc='z coefficient')

        # Initialize the sets
        self.__model__.I = Set(initialize=range(len(self.x)))
        self.__model__.J = Set(initialize=range(len(self.y[0])))
        self.__model__.L = Set(initialize=range(len(self.b[0])))

        # Initialize the variables
        self.__model__.alpha = Var(self.__model__.I, doc='alpha')
        self.__model__.gamma = Var(self.__model__.I,
                                  self.__model__.J,
                                  bounds=(0.0, None),
                                  doc='gamma')
        self.__model__.delta = Var(self.__model__.I, 
                                   self.__model__.L, 
                                   bounds=(0.0, None), 
                                   doc='delta')
        self.__model__.epsilon = Var(self.__model__.I, doc='residual')
        self.__model__.frontier = Var(self.__model__.I,
                                      bounds=(0.0, None),
                                      doc='estimated frontier')

        # Setup the objective function and constraints
        self.__model__.objective = Objective(rule=self.__objective_rule(),
                                             sense=minimize,
                                             doc='objective function')
        self.__model__.regression_rule = Constraint(self.__model__.I,
                                                    rule=self.__regression_rule(),
                                                    doc='regression equation')
        if self.cet == CET_MULT:
            self.__model__.log_rule = Constraint(self.__model__.I,
                                                 rule=self.__log_rule(),
                                                 doc='log-transformed regression equation')
        self.__model__.afriat_rule = Constraint(self.__model__.I,
                                                self.__model__.I,
                                                rule=self.__afriat_rule(),
                                                doc='afriat inequality')
        self.__model__.disposability_rule = Constraint(self.__model__.I,
                                                        self.__model__.I,
                                                        rule=self.__disposability_rule(),
                                                        doc='weak disposibility')

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0



    def __regression_rule(self):
        """Return the proper regression constraint"""
        if self.cet == CET_ADDI:
            if self.rts == RTS_VRS:
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return self.x[i] == - model.alpha[i] \
                                + sum(model.gamma[i, j] * self.y[i][j] for j in model.J) \
                                - sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                                - sum(model.lamda[k] * self.z[i][k] for k in model.K) \
                                - model.epsilon[i]

                    return regression_rule

                def regression_rule(model, i):
                    return self.x[i] == - model.alpha[i] \
                            + sum(model.gamma[i, j] * self.y[i][j] for j in model.J) \
                            - sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                            - model.epsilon[i]

                return regression_rule
            elif self.rts == RTS_CRS:
                if type(self.z) != type(None):
                    def regression_rule(model, i):
                        return self.x[i] == sum(model.gamma[i, j] * self.y[i][j] for j in model.J) \
                                - sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                                - sum(model.lamda[k] * self.z[i][k] for k in model.K) - model.epsilon[i]

                    return regression_rule

                def regression_rule(model, i):
                    return self.x[i] == sum(model.gamma[i, j] * self.y[i][j] for j in model.J) \
                            - sum(model.delta[i, l] * self.b[i][l] for l in model.L) \
                            - model.epsilon[i]

                return regression_rule

        elif self.cet == CET_MULT:
            if type(self.z) != type(None):
                def regression_rule(model, i):
                    return log(self.x[i]) == - log(model.frontier[i] + 1) \
                            - sum(model.lamda[k] * self.z[i][k] for k in model.K) - model.epsilon[i]

                return regression_rule

            def regression_rule(model, i):
                return log(self.x[i]) == - log(model.frontier[i] + 1) - model.epsilon[i]

            return regression_rule

        raise ValueError("Undefined model parameters.")

    def __log_rule(self):
        """Return the proper log constraint"""
        if self.cet == CET_MULT:
            if self.rts == RTS_VRS:

                def log_rule(model, i):
                    return model.frontier[i] == model.alpha[i] - sum(
                        model.gamma[i, j] * self.y[i][j] for j in model.J) \
                            + sum(model.delta[i, l] * self.b[i][l] for l in model.L) - 1

                return log_rule
            elif self.rts == RTS_CRS:

                def log_rule(model, i):
                    return model.frontier[i] == - sum(
                        model.gamma[i, j] * self.y[i][j] for j in model.J) \
                            + sum(model.delta[i, l] * self.b[i][l] for l in model.L) - 1

                return log_rule

        raise ValueError("Undefined model parameters.")

    def __afriat_rule(self):  ### todo 改变 afriat rule
        """Return the proper afriat inequality constraint"""
        if self.fun == FUN_PROD:
            __operator = NumericValue.__le__
        elif self.fun == FUN_COST:
            __operator = NumericValue.__ge__


        if self.rts == RTS_VRS:

            def afriat_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return __operator(
                    model.alpha[i] - sum(model.gamma[i, j] * self.y[i][j] for j in model.J)
                        + sum(model.delta[i, l] * self.b[i][l] for l in model.L),
                    model.alpha[h] - sum(model.gamma[h, j] * self.y[i][j] for j in model.J)
                        + sum(model.delta[h, l] * self.b[i][l] for l in model.L) )

            return afriat_rule
        elif self.rts == RTS_CRS:

            def afriat_rule(model, i, h):
                if i == h:
                    return Constraint.Skip
                return __operator(
                    - sum(model.gamma[i, j] * self.y[i][j] for j in model.J)
                        + sum(model.delta[i, l] * self.b[i][l] for l in model.L),
                    - sum(model.gamma[h, j] * self.y[i][j] for j in model.J)
                        + sum(model.delta[h, l] * self.b[i][l] for l in model.L))

            return afriat_rule


        raise ValueError("Undefined model parameters.")

    def __disposability_rule(self):
        """Return the proper weak disposability constraint"""
        def disposability_rule(model, i, h):
            if i == h:
                return Constraint.Skip
            return model.alpha[i] + sum( self.x[h][j] for j in model.J) >= 0
        
        return disposability_rule
        


    def display_gamma(self):
        """Display beta value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_desirable_output(self.y)
        self.__model__.gamma.display()






    def get_gamma(self):
        """Return beta value by array"""
        tools.assert_optimized(self.optimization_status)
        gamma = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.gamma),
                                                          list(self.__model__.gamma[:, :].value))])
        gamma = pd.DataFrame(gamma, columns=['Name', 'Key', 'Value'])
        gamma = gamma.pivot(index='Name', columns='Key', values='Value')
        return gamma.to_numpy()





    def get_frontier(self):
        """Return estimated frontier value by array"""
        tools.assert_optimized(self.optimization_status)
        if self.cet == CET_MULT and type(self.z) == type(None):
            frontier = np.asarray(list(self.__model__.frontier[:].value)) + 1
        elif self.cet == CET_MULT and type(self.z) != type(None):
            frontier = list(np.divide(1, np.exp(
                self.get_residual() + self.get_lamda() * np.asarray(self.z)[:, 0] ) * self.x ) - 1)
        elif self.cet == CET_ADDI:
            frontier = np.asarray(self.y) - self.get_residual()
        return np.asarray(frontier)

