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
import jax.numpy as jnp


class Dynamics:
    def setupDynamics(
        self,
    ) -> None:

        # A and B matrices constant
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([0, 1])

        # Integration options
        self.int_options = {"rtol": 1e-9, "atol": 1e-9}

        # Simulation data
        self.del_t = 0.02  # [sec]
        self.total_steps = int(7 / self.del_t) + 2
        self.curr_step = 0

        # Initial conditions
        self.x0 = np.array([-2, 0.5])

        self.sup_fcl = 0.2

    def propMain(self, t, x, u, args):
        """
        Propagation function for dynamics and STM if applicable.
        Could be optimized (linear system).

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) * u
        if len(x) > lenx:
            # Construct F
            F = self.computeJacobianSTM(x[:lenx])

            # Extract STM & reshape
            STM = x[lenx:].reshape(lenx, lenx)
            dSTM = F @ STM

            # Reshape back to column
            dSTM = dSTM.reshape(lenx**2)
            dx[lenx:] = dSTM

        return dx

    def computeJacobianSTM(self, x):
        """
        Compute Jacobian of dynamics.
        """
        jac = self.A
        return jac

    def f_x(self, x):
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        f = self.A @ x
        return f

    def g_x(self, x):
        """
        Function g(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        g = self.B
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
        dx[:lenx] = self.A @ x[:lenx] + self.B * self.backupControl(x[:lenx])

        if len(x) > lenx:
            # Construct F
            F = self.A

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
        dx[:lenx] = self.A @ x[:lenx] + self.B * self.backupControl(x[:lenx])

        # Construct F (closed-loop Jacobian)
        F = self.A

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
