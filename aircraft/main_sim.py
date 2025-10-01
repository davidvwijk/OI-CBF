"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for optimally interpolated safe controllers.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module runs full simulation for aircraft example.

"""

import numpy as np
from safety import ASIF
from control import Control
from dynamics import Dynamics


class Simulation(ASIF, Control, Dynamics):
    def __init__(
        self,
        safety_flag=True,
        verbose=True,
        bCBF_QP=True,
        regFun=False,
        OI_QP_flag=False,
    ) -> None:

        self.setupDynamics()
        self.setupControl()
        self.setupASIF()

        self.verbose = verbose
        self.safety_flag = safety_flag
        self.bCBF_QP = bCBF_QP
        self.regFun = regFun
        self.OI_QP_flag = OI_QP_flag

    def sim(self):
        """
        Simulates trajectory for pre-specified number of timesteps and performs
        point-wise safety-critical control modifications if applicable.

        """
        x0 = self.x0
        total_steps = self.total_steps

        # Tracking variables
        x_full = np.zeros((len(x0), total_steps))
        u_act_full = np.zeros((len(self.u_bounds), total_steps))
        u_des_full = np.zeros((len(self.u_bounds), total_steps))
        solver_times, cb_times, intervened = [], [], []
        x_full[:, 0] = x0
        x_curr = x0

        # Main loop
        for i in range(1, total_steps):
            t = self.curr_step * self.del_t

            # Generate desired control
            u_des = self.primaryControl(x_curr)
            u_des_full[:, i] = u_des

            # Modify desired control using safety filter
            if self.safety_flag:
                if self.bCBF_QP:
                    u, boolean, sdt, cdt = self.vanilla_bCBF(x_curr, u_des)
                elif self.regFun:
                    u, boolean, sdt, cdt = self.vanilla_blending(x_curr, u_des)
                elif self.OI_QP_flag:
                    u, boolean, sdt, cdt = self.OI_QP(x_curr, u_des)

                solver_times.append(sdt)
                cb_times.append(cdt)
                if boolean:
                    intervened.append(i)
            else:
                u = u_des

            u_act_full[:, i] = u

            # Propagate states with control
            x_curr = self.integrateState(
                x_curr,
                u,
                self.del_t,
                self.int_options,
            )
            x_full[:, i] = x_curr

            self.curr_step += 1

        avg_solver_t = []
        max_solver_t = []

        return (
            x_full,
            total_steps,
            u_des_full,
            u_act_full,
            intervened,
            avg_solver_t,
            max_solver_t,
        )


if __name__ == "__main__":

    env = Simulation(
        safety_flag=True,
        verbose=True,
        bCBF_QP=False,
        regFun=False,
        OI_QP_flag=True,
    )
    print(
        "Running simulation with parameters:",
        "Safety:",
        env.safety_flag,
        "bCBF_QP:",
        env.bCBF_QP,
        "Reg. function:",
        env.regFun,
        "OI-QP:",
        env.OI_QP_flag,
    )

    (
        x_full,
        total_steps,
        u_des_full,
        u_act_full,
        intervened,
        avg_solver_t,
        max_solver_t,
    ) = env.sim()
