# import dependencies
import numpy as np
import pandas as pd
from .utils import CNLSDDFG1,CNLSDDFG2, CNLSDDFZG1, CNLSDDFZG2, sweet, tools
from .constant import CET_ADDI, FUN_PROD, FUN_COST, OPT_DEFAULT, RTS_CRS, RTS_VRS,OPT_LOCAL
import time





class CNLSDDFG:
    """Convex Nonparametric Least Square (CNLSDDF) and Genetic algorithm
    """
    def __init__(self, y, x, b, z=None, gy=[1], gx=[1], gb=[1], fun=FUN_PROD, rts=RTS_VRS):
        """CNLSDDFG model

        Args:
            y (ndarray): output variable.
            x (ndarray): input variables.
            b (ndarray, optional): undersiable variables.
            z (ndarray, optional): Contextual variable(s). Defaults to None.
            gy (list, optional): output directional vector. Defaults to [1].
            gx (list, optional): input directional vector. Defaults to [1].
            gb (list, optional): undesirable output directional vector. Defaults to [1].
            fun (String, optional): FUN_PROD (production frontier) or FUN_COST (cost frontier). Defaults to FUN_PROD.
            rts (String, optional): RTS_VRS (variable returns to scale) or RTS_CRS (constant returns to scale). Defaults to RTS_VRS.
        """
        # TODO(error/warning handling): Check the configuration of the model exist
        self.cutactive = sweet.sweet(np.column_stack(x,b))
        self.y, self.x, self.b, self.z,self.gy, self.gx, self.gb = \
            tools.assert_valid_direciontal_data_with_z(y,x,b,z,gy,gx,gb)


        self.fun = fun
        self.rts = rts

        # active (added) violated concavity constraint by iterative procedure
        self.active = np.zeros((len(x), len(x)))
        # violated concavity constraint
        self.active2 = np.zeros((len(x), len(x)))

        # active (added) violated concavity constraint for weak disposbility constrains by iterative procedure
        self.activeweak = np.zeros((len(x), len(x)))
        # violated concavity constraint for weak disposbility constrains
        self.activeweak2 = np.zeros((len(x), len(x)))

        # Optimize model
        self.optimization_status = 0
        self.problem_status = 0



    def optimize(self, email=OPT_LOCAL, solver=OPT_DEFAULT):
        """Optimize the function by requested method"""
        # TODO(error/warning handling): Check problem status after optimization
        self.t0 = time.time()
        if type(self.z) != type(None):
            model1 = CNLSDDFZG1.CNLSDDFZG1(
                self.y, self.x, self.b, self.z, self.cutactive, self.gy, self.gx, self.gb, self.fun, self.rts)
        else:
            model1 = CNLSDDFG1.CNLSDDFG1(
                self.y, self.x, self.b, self.cutactive, self.gy, self.gx, self.gb, self.fun, self.rts)
        model1.optimize(email, solver)
        self.alpha = model1.get_alpha()
        self.beta = model1.get_beta()
        self.gamma = model1.get_gamma()
        self.delta = model1.get_delta()
        self.__model__ = model1.__model__

        self.count = 0
        while self.__convergence_test(self.alpha, self.beta, self.gamma, self.delta) > 0.0001 :
            if type(self.z) != type(None):
                model2 = CNLSDDFZG2.CNLSDDFZG2(
                    self.y, self.x, self.b, self.z,self.cutactive, self.active,
                    self.gy, self.gx, self.gb,  self.fun, self.rts)
            else:
                model2 = CNLSDDFG2.CNLSDDFG2(
                    self.y, self.x, self.b, self.cutactive, self.active,
                    self.gy, self.gx, self.gb, self.fun, self.rts)
            model2.optimize(email, solver)
            self.alpha = model2.get_alpha()
            self.beta = model2.get_beta()
            self.gamma = model2.get_gamma()
            self.delta = model2.get_delta()
            # TODO: Replace print with log system
            # print("Genetic Algorithm Convergence : %8f" %
            #       (self.__convergence_test(self.alpha, self.beta)))
            self.__model__ = model2.__model__
            self.count += 1
        self.optimization_status = 1
        self.tt = time.time() - self.t0

    def __convergence_test(self, alpha, beta, gamma, delta):
        x = np.asarray(self.x)
        y = np.asarray(self.y)
        b = np.asarray(self.b)
        activetmp1 = 0.0
        if self.rts == RTS_VRS and self.fun == FUN_PROD:
        # go into the loop
            for i in range(len(x)):
                activetmp = 0.0
                # go into the sub-loop and find the violated concavity constraints
                for j in range(len(x)):
                    self.active2[i, j] = alpha[i] + np.sum(beta[i, :] * x[i, :]) \
                        + np.sum(delta[i, :] * b[i, :]) - np.sum(gamma[i, :] * y[i, :]) \
                                         - alpha[j] - np.sum(beta[j, :] * x[i, :]) \
                        - np.sum(delta[j, :] * b[i, :] + np.sum(gamma[j, :] * y[i, :]))

                    if self.active2[i, j] > activetmp:
                        activetmp = self.active2[i, j]

            # find the maximal violated constraint in sub-loop and added into the active matrix
                for j in range(len(x)):
                    if self.active2[i, j] >= activetmp and activetmp > 0:
                        self.active[i, j] = 1
                if activetmp > activetmp1:
                    activetmp1 = activetmp


        elif self.rts == RTS_VRS and self.fun == FUN_COST:
        # go into the loop
            for i in range(len(x)):
                activetmp = 0.0
                # go into the sub-loop and find the violated concavity constraints
                for j in range(len(x)):
                    self.active2[i, j] = -alpha[i] - np.sum(beta[i, :] * x[i, :]) \
                        - np.sum(delta[i, :] * b[i, :]) + np.sum(gamma[i, :] * y[i, :]) \
                                         + alpha[j] + np.sum(beta[j, :] * x[i, :]) \
                        + np.sum(delta[j, :] * b[i, :] - np.sum(gamma[j, :] * y[i, :]))

                    if self.active2[i, j] > activetmp:
                        activetmp = self.active2[i, j]

            # find the maximal violated constraint in sub-loop and added into the active matrix
                for j in range(len(x)):
                    if self.active2[i, j] >= activetmp and activetmp > 0:
                        self.active[i, j] = 1
                if activetmp > activetmp1:
                    activetmp1 = activetmp


        elif self.rts == RTS_CRS and self.fun == FUN_PROD:
        # go into the loop
            for i in range(len(x)):
                activetmp = 0.0
                # go into the sub-loop and find the violated concavity constraints
                for j in range(len(x)):
                    self.active2[i, j] = np.sum(beta[i, :] * x[i, :]) \
                        + np.sum(delta[i, :] * b[i, :]) - np.sum(gamma[i, :] * y[i, :]) \
                                          - np.sum(beta[j, :] * x[i, :]) \
                        - np.sum(delta[j, :] * b[i, :] + np.sum(gamma[j, :] * y[i, :]))

                    if self.active2[i, j] > activetmp:
                        activetmp = self.active2[i, j]

            # find the maximal violated constraint in sub-loop and added into the active matrix
                for j in range(len(x)):
                    if self.active2[i, j] >= activetmp and activetmp > 0:
                        self.active[i, j] = 1
                if activetmp > activetmp1:
                    activetmp1 = activetmp

        elif self.rts == RTS_CRS and self.fun == FUN_COST:
        # go into the loop
            for i in range(len(x)):
                activetmp = 0.0
                # go into the sub-loop and find the violated concavity constraints
                for j in range(len(x)):
                    self.active2[i, j] = - np.sum(beta[i, :] * x[i, :]) \
                        - np.sum(delta[i, :] * b[i, :]) + np.sum(gamma[i, :] * y[i, :]) \
                                          + np.sum(beta[j, :] * x[i, :]) \
                        + np.sum(delta[j, :] * b[i, :] - np.sum(gamma[j, :] * y[i, :]))
                    if self.active2[i, j] > activetmp:
                        activetmp = self.active2[i, j]

            # find the maximal violated constraint in sub-loop and added into the active matrix
                for j in range(len(x)):
                    if self.active2[i, j] >= activetmp and activetmp > 0:
                        self.active[i, j] = 1
                if activetmp > activetmp1:
                    activetmp1 = activetmp
        return activetmp



    def display_status(self):
        """Display the status of problem"""
        tools.assert_optimized(self.optimization_status)
        print(self.display_status)

    def display_alpha(self):
        """Display alpha value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_various_return_to_scale(self.rts)
        self.__model__.alpha.display()

    def display_beta(self):
        """Display beta value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.beta.display()


    def display_lamda(self):
        """Display lamda value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_contextual_variable(self.z)
        self.__model__.lamda.display()

    def display_residual(self):
        """Dispaly residual value"""
        tools.assert_optimized(self.optimization_status)
        self.__model__.epsilon.display()

    def display_delta(self):
        """Display delta value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_undesirable_output(self.b)
        self.__model__.delta.display()

    def display_gamma(self):
        """Display gamma value"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_desirable_output(self.y)
        self.__model__.gamma.display()

    def get_status(self):
        """Return status"""
        return self.optimization_status

    def get_alpha(self):
        """Return alpha value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_various_return_to_scale(self.rts)
        alpha = list(self.__model__.alpha[:].value)
        return np.asarray(alpha)

    def get_beta(self):
        """Return beta value by array"""
        tools.assert_optimized(self.optimization_status)
        beta = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.beta),
                                                          list(self.__model__.beta[:, :].value))])
        beta = pd.DataFrame(beta, columns=['Name', 'Key', 'Value'])
        beta = beta.pivot(index='Name', columns='Key', values='Value')
        return beta.to_numpy()

    def get_residual(self):
        """Return residual value by array"""
        tools.assert_optimized(self.optimization_status)
        residual = list(self.__model__.epsilon[:].value)
        return np.asarray(residual)

    def get_lamda(self):
        """Return beta value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_contextual_variable(self.z)
        lamda = list(self.__model__.lamda[:].value)
        return np.asarray(lamda)

    def get_frontier(self):
        """Return estimated frontier value by array"""
        tools.assert_optimized(self.optimization_status)
        frontier = np.asarray(self.y) - self.get_residual()
        return np.asarray(frontier)

    def get_delta(self):
        """Return delta value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_undesirable_output(self.b)
        delta = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.delta),
                                                           list(self.__model__.delta[:, :].value))])
        delta = pd.DataFrame(delta, columns=['Name', 'Key', 'Value'])
        delta = delta.pivot(index='Name', columns='Key', values='Value')
        return delta.to_numpy()

    def get_gamma(self):
        """Return gamma value by array"""
        tools.assert_optimized(self.optimization_status)
        tools.assert_desirable_output(self.y)
        gamma = np.asarray([i + tuple([j]) for i, j in zip(list(self.__model__.gamma),
                                                           list(self.__model__.gamma[:, :].value))])
        gamma = pd.DataFrame(gamma, columns=['Name', 'Key', 'Value'])
        gamma = gamma.pivot(index='Name', columns='Key', values='Value')
        return gamma.to_numpy()