"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for optimally interpolated safe controllers.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module containing control laws.

"""

import numpy as np


class Control:
    def setupControl(
        self,
    ) -> None:

        # Control limits
        self.up_max = np.pi / 2  # [rad/sec]
        self.up_min = -np.pi / 2  # [rad/sec]
        self.uz_min = -1
        self.uz_max = 4
        self.u_bounds = (
            (self.up_min, self.up_max),
            (self.uz_min, self.uz_max),
        )
        self.lenu = len(self.u_bounds)

        # Gains
        self.kphi = 3
        self.kP = 2
        self.kN = 0.4
        self.ktheta = 0.65
        self.kpsi = 0.45
        self.kH = 0.0001

        # Setpoint for backup
        self.H_star = 10000

        # Setpoint for primary
        self.pN_primary = 27000
        self.pE_primary = 0
        self.H_primary = 10000

        # Limits for aircraft
        self.phimax = np.deg2rad(45)

    def primaryControl(self, x):
        """
        Primary controller producing desired control at each step.

        """
        phi, theta, psi, pN, pE, H, P, Nz = x

        eN = self.pN_primary - pN
        eE = self.pE_primary - pE
        psi_los = np.arctan2(eE, eN)
        epsi = np.arctan2(np.sin(psi_los - psi), np.cos(psi_los - psi))

        # Take absolute value so aircraft keeps turning into geofence
        omega_d = self.kpsi * np.abs(epsi)
        phi_d = self.phimax * np.tanh((self.VT / self.g) * omega_d / self.phimax)
        Nz_star = 1.0 / np.cos(phi_d)
        up = P + self.tp * (self.kphi * (phi_d - phi) - self.kP * P)
        uz = Nz + self.tz * (
            self.kN * (Nz_star - Nz)
            - self.ktheta * theta
            + self.kH * (self.H_primary - H)
        )
        return np.array(
            [
                np.clip(up, self.up_min, self.up_max),
                np.clip(uz, self.uz_min, self.uz_max),
            ]
        )

    def backupControl(self, x):
        """
        Safe backup controller.

        """
        phi, theta, psi, pN, pE, H, P, Nz = x

        phi_d = self.phi_star
        Nz_star = 1.0 / np.cos(phi_d)
        up = P + self.tp * (self.kphi * (phi_d - phi) - self.kP * P)
        uz = Nz + self.tz * (
            self.kN * (Nz_star - Nz) - self.ktheta * theta + self.kH * (self.H_star - H)
        )

        # Saturate softplus
        A = self.up_max
        s0_up = np.tanh(0.5 * 30 * A)
        gamma_up = 1 / (s0_up + 1e-6)
        up = (
            -A
            + np.log(1 + np.exp(30 * (gamma_up * up + A))) / 30
            - np.log(1 + np.exp(30 * (gamma_up * up - A))) / 30
        )
        L = self.uz_min
        U = self.uz_max
        v0 = (1 / 30) * (np.log(1 + np.exp(30 * L)) - np.log(1 + np.exp(-30 * U)))
        s0 = 1 / (1 + np.exp(-30 * (0 - v0 - L))) - 1 / (1 + np.exp(-30 * (0 - v0 - U)))
        gamma = 1 / (s0 + 1e-6)
        uz = (
            L
            + np.log(1 + np.exp(30 * (gamma * uz - v0 - L))) / 30
            - np.log(1 + np.exp(30 * (gamma * uz - v0 - U))) / 30
        )
        return np.array([up, uz])
