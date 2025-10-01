"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for optimally interpolated safe controllers.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module contains functions required for propagating dynamics.

"""

import numpy as np
from scipy.integrate import solve_ivp


class Dynamics:
    def setupDynamics(
        self,
    ) -> None:
        # Integration options
        self.int_options = {"rtol": 1e-6, "atol": 1e-6}

        # Simulation data
        self.del_t = 0.15  # [sec]
        self.total_steps = int(110 / self.del_t) + 2
        self.curr_step = 0

        # Initial conditions
        # x = [phi, theta, psi, pN, pE, H, P, Nz]
        self.x0 = np.array([0, 0, 0, 16000, 0, 10000, 0, 1])
        self.lenx = len(self.x0)

        # Constants
        self.g = 9.81  # [m/s^2]
        self.tp = 1  # [s]
        self.tz = 1  # [s]
        self.VT = 200  # [m/s]

        # Discretization constant
        self.sup_fcl = 150

    def propMain(self, t, x, u, args):
        """
        Propagation function for dynamics and STM if applicable.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) @ u

        return dx

    def computeJacobianCL(self, x):
        """
        Compute Jacobian of dynamics. Converted from MATLAB.
        """
        phi, theta, psi, pN, pE, H, P, Nz = x

        # Constants from self
        phimax = self.phi_star
        kP = self.kP
        tp = self.tp
        kN = self.kN
        tz = self.tz
        ktheta = self.ktheta
        H_star = self.H_star
        VT = self.VT
        up_max = self.up_max
        uz_min = self.uz_min
        uz_max = self.uz_max
        kH = self.kH
        kphi = self.kphi
        g = self.g

        t2 = np.cos(phi)
        t3 = np.cos(phimax)
        t4 = np.cos(psi)
        t5 = np.cos(theta)
        t6 = np.sin(phi)
        t7 = np.sin(psi)
        t8 = np.sin(theta)
        t9 = np.tan(theta)
        t10 = P * kP
        t11 = kP * tp
        t12 = kN * tz
        t13 = ktheta * theta
        t14 = -H_star
        t15 = 1.0 / VT
        t16 = -phimax
        t17 = 1.0 / tp
        t18 = 1.0 / tz
        t19 = up_max * 1.5e1
        t20 = up_max * 3.0e1
        t21 = uz_min * 3.0e1
        t22 = uz_max * 3.0e1
        t23 = 1.0 / t3
        t24 = 1.0 / t5
        t25 = np.exp(t21)
        t26 = np.tanh(t19)
        t27 = -t20
        t28 = -t21
        t29 = -t22
        t31 = H + t14
        t32 = t11 - 1.0
        t33 = t12 - 1.0
        t34 = phi + t16
        t30 = np.exp(t29)
        t35 = t25 + 1.0
        t37 = kH * t31
        t38 = kphi * t34
        t39 = -t23
        t48 = t26 + 1.0e-6
        t36 = np.log(t35)
        t40 = t30 + 1.0
        t42 = Nz + t39
        t46 = t10 + t38
        t50 = 1.0 / t48
        t41 = np.log(t40)
        t43 = -t36
        t44 = kN * t42
        t47 = t46 * tp
        t45 = -t41
        t49 = -t47
        t52 = t13 + t37 + t44
        t51 = P + t49
        t53 = t21 + t36 + t45
        t54 = t22 + t36 + t45
        t57 = t52 * tz
        t55 = np.exp(t53)
        t56 = np.exp(t54)
        t60 = -t57
        t65 = t50 * t51 * 3.0e1
        t58 = t55 + 1.0
        t59 = t56 + 1.0
        t61 = Nz + t60
        t66 = t20 + t65
        t67 = t27 + t65
        t62 = 1.0 / t58
        t63 = 1.0 / t59
        t68 = np.exp(t66)
        t69 = np.exp(t67)
        t64 = -t63
        t70 = t68 + 1.0
        t71 = t69 + 1.0
        t72 = 1.0 / t70
        t73 = 1.0 / t71
        t74 = t62 + t64 + 1.0e-6
        t75 = 1.0 / t74
        t76 = t61 * t75 * 3.0e1
        t77 = t28 + t41 + t43 + t76
        t78 = t29 + t41 + t43 + t76
        t79 = np.exp(t77)
        t80 = np.exp(t78)
        t81 = t79 + 1.0
        t82 = t80 + 1.0
        t83 = 1.0 / t81
        t84 = 1.0 / t82
        mt1 = [
            Nz * g * t2 * t9 * t15,
            -Nz * g * t6 * t15,
            Nz * g * t2 * t15 * t24,
            0.0,
            0.0,
            0.0,
            -t17 * (kphi * t50 * t68 * t72 * tp - kphi * t50 * t69 * t73 * tp),
            0.0,
            Nz * g * t6 * t15 * (t9**2 + 1.0),
            g * t8 * t15,
            Nz * g * t6 * t8 * t15 * t24**2,
            -VT * t4 * t8,
            -VT * t7 * t8,
            VT * t5,
            0.0,
            -t18 * (ktheta * t75 * t79 * t83 * tz - ktheta * t75 * t80 * t84 * tz),
            0.0,
            0.0,
            0.0,
            -VT * t5 * t7,
            VT * t4 * t5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -t18 * (kH * t75 * t79 * t83 * tz - kH * t75 * t80 * t84 * tz),
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -t17 * (t32 * t50 * t68 * t72 - t32 * t50 * t69 * t73 + 1.0),
            0.0,
            g * t6 * t9 * t15,
            g * t2 * t15,
            g * t6 * t15 * t24,
            0.0,
            0.0,
            0.0,
        ]
        mt2 = [0.0, -t18 * (t33 * t75 * t79 * t83 - t33 * t75 * t80 * t84 + 1.0)]

        J = np.reshape(mt1 + mt2, (8, 8), order="F")
        J = np.real_if_close(J)

        return J

    def f_x(self, x):
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.
        x = [phi, theta, psi, pN, pE, H, P, Nz]

        """
        f = np.array(
            [
                x[6] + (x[7] * self.g / self.VT) * np.sin(x[0] * np.tan(x[1])),
                self.g / self.VT * (x[7] * np.cos(x[0]) - np.cos(x[1])),
                (x[7] * self.g * np.sin(x[0])) / (self.VT * np.cos(x[1])),
                self.VT * np.cos(x[1]) * np.cos(x[2]),
                self.VT * np.cos(x[1]) * np.sin(x[2]),
                self.VT * np.sin(x[1]),
                -1 / self.tp * x[6],
                -1 / self.tz * x[7],
            ]
        )
        return f

    def g_x(self, x):
        """
        Function g(x) for control affine dynamics, x_dot = f(x) + g(x)u.
        x = [phi, theta, psi, pN, pE, H, P, Nz]

        """
        g = np.zeros((self.lenx, 2))
        g[6:, :] = np.array([[1 / self.tp, 0], [0, 1 / self.tz]])
        return g

    def integrateState(self, x, u, t_step, options):
        """
        State integrator using propagation function.

        """
        t_step = (0.0, t_step)
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMain(t, x, u, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
        )
        x = soltn.y[:, -1]
        return x

    def propMainBackup(self, t, x, args):
        """
        Propagation function for backup dynamics and STM if applicable.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) @ self.backupControl(
            x[:lenx]
        )

        if len(x) > lenx:
            # Construct F
            F = self.computeJacobianCL(x[:lenx])

            # Extract STM & reshape
            STM = x[lenx:].reshape(lenx, lenx)
            dSTM = F @ STM

            # Reshape back to column
            dSTM = dSTM.reshape(lenx**2)
            dx[lenx:] = dSTM

        return dx

    def propMainBackupDirectional(self, t, x, args):
        """
        Propagation function for backup dynamics and directional derivatives [phi, qp, qb]

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) @ self.backupControl(
            x[:lenx]
        )

        # Construct F
        F = self.computeJacobianCL(x[:lenx])

        # Obtain directional derivative in direction of u_des
        dx[lenx : 2 * lenx] = F @ x[lenx : 2 * lenx]

        # Obtain directional derivative in direction of u_b
        dx[2 * lenx :] = F @ x[2 * lenx :]

        return dx

    def integrateStateBackup(self, x, tspan_b, options):
        """
        Propagate backup flow and sensitivity matrix (if applicable) over the backup horizon. Evaluate at discrete points.

        """
        t_step = (0.0, tspan_b[-1])
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMainBackup(t, x, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
            t_eval=tspan_b,
        )
        x = soltn.y[:, :]
        return x

    def integrateStateBackupDirectional(self, x, tspan_b, options):
        """
        Propagate backup flow and directional derivatives over the backup horizon. Evaluate at discrete points.

        """
        t_step = (0.0, tspan_b[-1])
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMainBackupDirectional(t, x, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
            t_eval=tspan_b,
        )
        x = soltn.y[:, :]
        return x
