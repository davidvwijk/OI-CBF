"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for optimally interpolated safe controllers.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module containing control laws.

"""


class Control:
    def setupControl(
        self,
    ) -> None:

        # Control limits
        self.u_max = 1
        self.u_bounds = ((-self.u_max, self.u_max),)

    def primaryControl(self, x_curr, t):
        """
        Primary controller producing desired control at each step.

        """
        u_des = self.u_max
        return u_des

    def backupControl(self, x):
        """
        Safe backup controller. Constant for this application.

        """
        u_b = -self.u_max
        return u_b
