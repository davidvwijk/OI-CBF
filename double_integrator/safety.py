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
import math
import quadprog
from scipy.integrate import quad


class Constraint:
    def alpha(self, x):
        """
        Strengthening function.

        """
        return 15 * x + x**3

    def alpha_2(self, x):
        """
        Strengthening function.

        """
        return 15 * x + x**3

    def alpha_b(self, x):
        """
        Strengthening function for reachability constraint.

        """
        return 10 * x

    def h1_x(self, x):
        """
        Safety constraint.

        """
        h = -x[0]
        return h

    def grad_h1(self, x):
        """
        Gradient of safety constraint.

        """
        g = np.array([-1, 0])
        return g

    def hb_x(self, x):
        """
        Reachability constraint.

        """
        hb = -x[1]
        return hb

    def grad_hb(self, x):
        """
        Gradient of reachability constraint.

        """
        gb = np.array([0, -1])
        return gb

    def h_I(self, phi):
        """
        Closest point to boundary of h or hb.

        """
        return np.min(np.append(self.h1_x(phi[:, :].T), self.hb_x(phi[-1, :].T)))


class ASIF(Constraint):
    def setupASIF(
        self,
    ) -> None:

        # Backup properties
        self.backupTime = 1  # [sec] (total backup time)
        self.tspan_b = np.arange(0, self.backupTime + self.del_t, self.del_t)
        self.backupTrajs = []
        self.backup_save_N = 5  # saves every N backup trajectory (for plotting)
        self.delta_array = [0]

        self.mu_array = []

        # Tightening constants
        self.Lh_const = 1

        # Regulation function tuning
        self.beta = 10

    def vanilla_bCBF(self, x, u_des):
        """
        Backup CBF QP.

        """

        # QP objective function
        M = np.eye(2)
        q = np.array(
            [u_des, 0.0]
        )  # Need to append the control with 0 to get at least 2 dimensions

        # Control constraints
        G = [[1.0, 0.0], [-1.0, 0.0]]
        h = [-self.u_max, -self.u_max]

        # Backup trajectory points
        rtapoints = len(self.tspan_b)

        # State tracking array
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x

        # Sensitivity matrix tracking array
        S = np.zeros((lenx, lenx, rtapoints))
        S[:, :, 0] = np.eye(lenx)

        # Simulate flow under backup control law
        new_x = np.concatenate((x, S[:, :, 0].flatten()))

        backupFlow = self.integrateStateBackup(
            new_x,
            self.tspan_b,
            self.int_options,
        )

        phi[:, :] = backupFlow[:lenx, :].T
        S[:, :, :] = backupFlow[lenx:, :].reshape(lenx, lenx, rtapoints)

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        fx_0 = self.f_x(x)
        gx_0 = self.g_x(x)

        # Construct barrier constraint for each point along trajectory
        for i in range(
            1, rtapoints
        ):  # Skip first point because of relative degree issue

            h_phi = self.h1_x(phi[i, :])
            gradh_phi = self.grad_h1(phi[i, :])
            g_temp_i = gradh_phi.T @ S[:, :, i] @ gx_0

            # Discretization tightening constant
            mu_d = (self.del_t / 2) * self.Lh_const * self.sup_fcl

            h_temp_i = -(gradh_phi @ S[:, :, i] @ fx_0 + self.alpha(h_phi - mu_d))

            # Append constraint
            G.append([g_temp_i, 0])
            h.append(h_temp_i)

            # Make sure last point is in the backup set
            if i == rtapoints - 1:

                hb_phi = self.hb_x(phi[i, :])
                gradhb_phi = self.grad_hb(phi[i, :])

                h_temp_i = -(gradhb_phi @ S[:, :, i] @ fx_0 + self.alpha_b(hb_phi))
                g_temp_i = gradhb_phi.T @ S[:, :, i] @ gx_0

                # Append constraint
                G.append([g_temp_i, 0])
                h.append(h_temp_i)

        # Solve QP
        try:
            tic = time.perf_counter()
            sltn = quadprog.solve_qp(M, q, np.array(G).T, np.array(h), 0)
            u_act = sltn[0]
            active_constraint = sltn[5]
            toc = time.perf_counter()
            solver_dt = toc - tic
            u_act = u_act[0]  # Only extract scalar we need
        except:
            u_act = -1
            solver_dt = None
            if self.verbose:
                print("no soltn")

        # If safe action is different the desired action, filter is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solver_dt

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

        backupFlow = self.integrateStateBackup(
            new_x,
            self.tspan_b,
            self.int_options,
        )

        phi[:, :] = backupFlow[:lenx, :].T

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        h_I_plus = max(self.h_I(phi), 0)
        mu = np.exp(-self.beta * h_I_plus)
        u_act = (1 - mu) * u_des + mu * self.backupControl(x)

        self.mu_array.append(mu)

        # If safe action is different the desired action, filter is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, 0

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

        q_up[0, :] = fx_0 + gx_0 * u_des
        q_ub[0, :] = fx_0 + gx_0 * self.backupControl(x)

        # Simulate flow under backup control law
        new_x = np.concatenate((x, q_up[0, :], q_ub[0, :]))

        backupFlow = self.integrateStateBackupDirectional(
            new_x,
            self.tspan_b,
            self.int_options,
        )

        phi[:, :] = backupFlow[:lenx, :].T
        q_up[:, :] = backupFlow[lenx : 2 * lenx, :].T
        q_ub[:, :] = backupFlow[2 * lenx :, :].T

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        # Construct barrier constraint for each point along trajectory
        for i in range(1, rtapoints):  # Skipping first point bc. relative degree
            # Gradients
            h_phi = self.h1_x(phi[i, :])
            gradh_phi = self.grad_h1(phi[i, :])

            # Compute total derivatives
            hdot_up = gradh_phi @ q_up[i, :]
            hdot_ub = gradh_phi @ q_ub[i, :]

            # QP Constraints
            # Discretization tightening constant
            mu_d = (self.del_t / 2) * self.Lh_const * self.sup_fcl
            g_temp_i = hdot_ub - hdot_up
            h_temp_i = -(hdot_up + self.alpha(h_phi - mu_d))

            # Append constraint
            G.append([g_temp_i, 0])
            h.append(h_temp_i)

            # Make sure last point is in the backup set
            if i == rtapoints - 1:

                hb_phi = self.hb_x(phi[i, :])
                gradhb_phi = self.grad_hb(phi[i, :])

                # Compute total derivatives
                hdot_up = gradhb_phi @ q_up[i, :]
                hdot_ub = gradhb_phi @ q_ub[i, :]

                # QP Constraints
                g_temp_i = hdot_ub - hdot_up
                h_temp_i = -(hdot_up + self.alpha_b(hb_phi))

                # Append constraint
                G.append([g_temp_i, 0])
                h.append(h_temp_i)

        # Solve QP
        I_plus = np.array(G).T[0] > 0
        mu_starKKT = max(np.array(h)[I_plus] / np.array(G).T[0][I_plus])
        self.mu_array.append(mu_starKKT)
        u_act = (1 - mu_starKKT) * u_des + mu_starKKT * self.backupControl(x)

        # If safe action is different the desired action, filter is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, 0
