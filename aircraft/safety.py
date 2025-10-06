"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for optimally interpolated safe controllers.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module contains functions required for safety-critical control using control barrier functions.

"""

import numpy as np
import time
import quadprog


class Constraint:

    # Strengthening functions
    def alpha(self, x):
        return 0.072 * x

    def alpha_b(self, x):
        return 0.5 * x

    # Geofence safety functions
    def h_x_geo(self, x):
        h = self.geo_ng @ (x[3:5] - self.geo_pg)
        return h

    def h_x_geo_array(self, x):
        h = self.geo_ng.T @ (x[3:5, :].T - self.geo_pg).T
        return h

    def grad_h_geo(self, x):
        g = np.array([0, 0, 0, self.geo_ng[0], self.geo_ng[1], 0, 0, 0])
        return g

    # Backup set functions
    def hb1_x(self, x):
        hb = self.c_phi**2 - (self.phi_star - x[0]) ** 2
        return hb

    def grad_hb1(self, x):
        gb = np.array([2 * (self.phi_star - x[0]), 0, 0, 0, 0, 0, 0, 0])
        return gb

    def hb2_x(self, x):
        hb = self.c_theta**2 - x[1] ** 2
        return hb

    def grad_hb2(self, x):
        gb = np.array([0, -2 * x[1], 0, 0, 0, 0, 0, 0])
        return gb

    def hb3_x(self, x):
        hb = 1 - ((self.H_star - x[5]) / self.c_H) ** 2
        return hb

    def grad_hb3(self, x):
        gb = np.array([0, 0, 0, 0, 0, (2 * (self.H_star - x[5])) / (self.c_H**2), 0, 0])
        return gb

    def hb4_x(self, x):
        hb = self.c_P**2 - x[6] ** 2
        return hb

    def grad_hb4(self, x):
        gb = np.array([0, 0, 0, 0, 0, 0, -2 * x[6], 0])
        return gb

    def hb5_x(self, x):
        hb = self.c_Nz**2 - (self.Nz_star - x[7]) ** 2
        return hb

    def grad_hb5(self, x):
        gb = np.array([0, 0, 0, 0, 0, 0, 0, 2 * (self.Nz_star - x[7])])
        return gb

    def hb6_x(self, x):
        hb = (
            self.h_x_geo(x)
            + self.rho * (self.geo_ng @ np.array([-np.sin(x[2]), np.cos(x[2])]) - 1)
            - self.c_6
        )
        return hb

    def grad_hb6(self, x):
        gb = self.grad_h_geo(x) + self.rho * np.array(
            [
                0,
                0,
                -self.geo_ng[0] * np.cos(x[2]) - self.geo_ng[1] * np.sin(x[2]),
                0,
                0,
                0,
                0,
                0,
            ]
        )
        return gb

    def hb_softmin(self, x, hbfuns, kappa):
        """
        Soft-min constraint.
        handles: array of function handles for constraints

        """
        exp_arg = 0
        for i in range(len(hbfuns)):
            hb_fun = hbfuns[i]
            exp_arg += np.exp(-kappa * hb_fun(x))

        return -np.log(exp_arg) / kappa

    def grad_hb_softmin(self, x, hbfuns, gradfuns, kappa):
        """
        Gradient of soft min for multiple constraints.

        """
        w_grad = np.zeros(self.lenx)
        w_hb = 0
        for i in range(len(hbfuns)):
            gradhb_fun = gradfuns[i]
            hb_fun = hbfuns[i]
            w_grad += np.exp(-kappa * hb_fun(x)) * gradhb_fun(x)
            w_hb += np.exp(-kappa * hb_fun(x))

        return w_grad / w_hb

    def h_I(self, phi, hbfuns, kappa):
        """
        Closest point to boundary of h or hb.

        """
        # Discretization tightening constant
        mu_d = (self.del_t_b / 2) * self.Lh_const * self.sup_fcl
        return np.min(
            np.append(
                self.h_x_geo_array(phi[:, :].T) - mu_d,
                self.hb_softmin(phi[-1, :].T, hbfuns, kappa),
            )
        )


class ASIF(Constraint):
    def setupASIF(
        self,
    ) -> None:

        # Backup properties
        self.backupTime = 60  # [sec] (total backup time)
        self.del_t_b = 0.5
        self.tspan_b = np.arange(0, self.backupTime + self.del_t_b, self.del_t_b)
        self.backupTrajs = []
        self.backup_save_N = 5  # saves every N backup trajectory (for plotting)
        self.delta_array = [0]

        # Regulation function tuning
        self.beta = 1000

        # Tracking mu
        self.mu_array = []

        # Geofence parameters
        self.geo_ng = np.array([-1, 0])
        self.geo_pg = np.array([25000.0, 0.0])  # [pN,pE]

        # Tightening constants
        self.Lh_const = np.linalg.norm(self.geo_ng)

        # Functions used in soft-min for hb(x) defining C_B
        self.kappa = 100
        self.hb_funs = [
            lambda x: self.hb1_x(x),
            lambda x: self.hb2_x(x),
            lambda x: self.hb3_x(x),
            lambda x: self.hb4_x(x),
            lambda x: self.hb5_x(x),
            lambda x: self.hb6_x(x),
        ]
        self.gradhb_funs = [
            lambda x: self.grad_hb1(x),
            lambda x: self.grad_hb2(x),
            lambda x: self.grad_hb3(x),
            lambda x: self.grad_hb4(x),
            lambda x: self.grad_hb5(x),
            lambda x: self.grad_hb6(x),
        ]

        # Terminal constraint parameters
        self.c_phi = np.deg2rad(15)  # [rad]
        self.c_theta = np.deg2rad(15)  # [rad]
        self.c_H = 400  # [m]
        self.c_P = np.deg2rad(15)  # [rad/sec]
        self.c_Nz = 0.1  # [g]
        self.c_6 = 5  # [m]

        # Values for backup set
        self.phi_star = -self.phimax
        self.Nz_star = 1 / (np.cos(self.phi_star))
        self.rho = (self.VT**2) / (self.g * np.tan(self.phi_star))

    def vanilla_bCBF(self, x, u_des):
        """
        Vanilla bCBF-QP.

        """

        # QP objective function
        M = np.eye(self.lenu)
        q = u_des

        # QP actuation constraints
        G = np.vstack((np.eye(self.lenu), -np.eye(self.lenu)))
        h = np.array([self.up_min, self.uz_min, -self.up_max, -self.uz_max])

        # Backup trajectory points
        rtapoints = len(self.tspan_b)

        # State tracking array
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x

        # Sensitivity matrix tracking array
        S = np.zeros((lenx, lenx, rtapoints))
        S[:, :, 0] = np.eye(lenx)

        # Simulate system under backup control law
        new_x = np.concatenate((x, S[:, :, 0].flatten()))

        tic = time.perf_counter()

        backupFlow = self.integrateStateBackup(
            new_x,
            self.tspan_b,
            self.int_options,
        )

        toc = time.perf_counter()
        backupComputeTime = toc - tic

        phi[:, :] = backupFlow[:lenx, :].T
        S[:, :, :] = backupFlow[lenx:, :].reshape(lenx, lenx, rtapoints)

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        fx_0 = self.f_x(x)
        gx_0 = self.g_x(x)

        # Construct constraints
        for i in range(1, rtapoints):
            # Compute h(phi)
            h_phi = self.h_x_geo(phi[i, :])
            gradh_phi = self.grad_h_geo(phi[i, :])
            g_temp_i = gradh_phi.T @ S[:, :, i] @ gx_0

            # Discretization tightening constant
            mu_d = (self.del_t_b / 2) * self.Lh_const * self.sup_fcl

            h_temp_i = -(gradh_phi @ S[:, :, i] @ fx_0 + self.alpha(h_phi - mu_d))

            if i == 1:
                g_temp = g_temp_i
                h_temp = h_temp_i
            else:
                g_temp = np.vstack([g_temp, g_temp_i])
                h_temp = np.vstack([h_temp, h_temp_i])

            # Make sure last point is in the backup set
            if i == rtapoints - 1:
                hb_phi = self.hb_softmin(phi[i, :], self.hb_funs, self.kappa)
                gradhb_phi = self.grad_hb_softmin(
                    phi[i, :], self.hb_funs, self.gradhb_funs, self.kappa
                )

                h_temp_i = -((gradhb_phi @ S[:, :, i]) @ fx_0 + self.alpha_b(hb_phi))
                g_temp_i = (gradhb_phi.T @ S[:, :, i]) @ gx_0

                # Append constraint
                g_temp = np.vstack([g_temp, g_temp_i])
                h_temp = np.vstack([h_temp, h_temp_i])

        # Append constraints
        G = np.vstack([G, g_temp])
        h = np.vstack([h.reshape((-1, 1)), h_temp])

        # Solve QP
        d = h.reshape((len(h),))
        try:
            tic = time.perf_counter()
            soltn = quadprog.solve_qp(M, q, G.T, d, 0)
            active_constraint = soltn[5]
            # print(active_constraint)
            u_act = soltn[0]
            toc = time.perf_counter()
            solverComputeTime = toc - tic
        except:
            u_act = self.backupControl(x)
            solverComputeTime = None
            if self.verbose:
                print("no soltn")

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solverComputeTime, backupComputeTime

    def vanilla_blending(self, x, u_des):
        """
        Regulation function-based blending controller.
        Does not need sensitivity matrix computation, still requires flow computation.

        """

        # Backup trajectory points
        rtapoints = len(self.tspan_b)

        # State tracking array
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x

        # Simulate flow under backup control law
        new_x = x
        tic = time.perf_counter()
        backupFlow = self.integrateStateBackup(
            new_x,
            self.tspan_b,
            self.int_options,
        )
        toc = time.perf_counter()
        backupComputeTime = toc - tic

        phi[:, :] = backupFlow[:lenx, :].T

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        h_I_plus = max(self.h_I(phi, self.hb_funs, self.kappa), 0)
        mu = np.exp(-self.beta * h_I_plus)
        u_act = (1 - mu) * u_des + mu * self.backupControl(x)
        self.mu_array.append(mu)

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, 0, backupComputeTime

    def OI_QP(self, x, u_des):
        """
        QP with blending.

        """

        # Control constraints
        G = [[1.0, 0.0], [-1.0, 0.0]]
        h = [0, -1]

        # Backup trajectory points
        rtapoints = len(self.tspan_b)

        # State tracking array
        lenx = len(self.x0)
        phi, q_up, q_ub = (
            np.zeros((rtapoints, lenx)),
            np.zeros((rtapoints, lenx)),
            np.zeros((rtapoints, lenx)),
        )
        phi[0, :] = x

        fx_0 = self.f_x(x)
        gx_0 = self.g_x(x)

        q_up[0, :] = fx_0 + gx_0 @ u_des
        q_ub[0, :] = fx_0 + gx_0 @ self.backupControl(x)

        # Simulate flow under backup control law
        new_x = np.concatenate((x, q_up[0, :], q_ub[0, :]))

        tic = time.perf_counter()

        backupFlow = self.integrateStateBackupDirectional(
            new_x,
            self.tspan_b,
            self.int_options,
        )

        toc = time.perf_counter()
        backupComputeTime = toc - tic

        phi[:, :] = backupFlow[:lenx, :].T
        q_up[:, :] = backupFlow[lenx : 2 * lenx, :].T
        q_ub[:, :] = backupFlow[2 * lenx :, :].T

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        # Construct barrier constraint for each point along trajectory
        for i in range(1, rtapoints):  # Skipping first point bc. relative degree
            # Gradients
            h_phi = self.h_x_geo(phi[i, :])
            gradh_phi = self.grad_h_geo(phi[i, :])

            # Compute total derivatives
            hdot_up = gradh_phi @ q_up[i, :]
            hdot_ub = gradh_phi @ q_ub[i, :]

            # QP Constraints
            # Discretization tightening constant
            mu_d = (self.del_t_b / 2) * self.Lh_const * self.sup_fcl
            g_temp_i = hdot_ub - hdot_up
            h_temp_i = -(hdot_up + self.alpha(h_phi - mu_d))

            # Append constraint
            G.append([g_temp_i, 0])
            h.append(h_temp_i)

            # Make sure last point is in the backup set
            if i == rtapoints - 1:
                hb_phi = self.hb_softmin(phi[i, :], self.hb_funs, self.kappa)
                gradhb_phi = self.grad_hb_softmin(
                    phi[i, :], self.hb_funs, self.gradhb_funs, self.kappa
                )

                # Compute total derivatives
                hdot_up = gradhb_phi @ q_up[i, :]
                hdot_ub = gradhb_phi @ q_ub[i, :]

                # QP Constraints
                g_temp_i = hdot_ub - hdot_up
                h_temp_i = -(hdot_up + self.alpha(hb_phi))

                # Append constraint
                G.append([g_temp_i, 0])
                h.append(h_temp_i)

        # Solve QP
        tic = time.perf_counter()
        I_plus = np.array(G).T[0] > 0
        mu_starKKT = max(np.array(h)[I_plus] / np.array(G).T[0][I_plus])
        toc = time.perf_counter()
        solver_dt = toc - tic
        u_act = (1 - mu_starKKT) * u_des + mu_starKKT * self.backupControl(x)
        self.mu_array.append(mu_starKKT)

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solver_dt, backupComputeTime
