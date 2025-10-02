"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for optimally interpolated safe controllers.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Compares vanilla bCBF-QP, vanilla regulation function bCBF, and optimally interpolated controller.

"""

from main_sim import Simulation
from plotting_paper_comparison import PaperPlotter
from pathlib import Path


if __name__ == "__main__":

    show_plots = False
    save_plots = True

    folder = Path("plots")
    folder.mkdir(parents=True, exist_ok=True)

    # Run bCBF-QP
    print("Running simulation with bCBF-QP")
    env1 = Simulation(
        safety_flag=True,
        verbose=False,
        bCBF_QP=True,
    )
    (
        x_full_bCBFQP,
        _,
        u_des_full,
        u_act_full_QP,
        _,
        _,
        _,
    ) = env1.sim()

    # Run vanilla regulation function bCBF
    print("Running simulation with vanilla regulation function")
    env2 = Simulation(
        safety_flag=True,
        verbose=False,
        bCBF_QP=False,
        regFun=True,
        OI_QP_flag=False,
    )
    (
        x_full_vanillaBlending,
        _,
        _,
        u_act_full_vanillaBlending,
        _,
        _,
        _,
    ) = env2.sim()

    # Run optimally interpolated QP
    print("Running simulation with optimally interpolated QP")
    env3 = Simulation(
        safety_flag=True,
        verbose=False,
        bCBF_QP=False,
        regFun=False,
        OI_QP_flag=True,
    )
    (
        x_full_OI_QP,
        _,
        _,
        u_act_full_OI_QP,
        _,
        _,
        _,
    ) = env3.sim()

    PaperPlotter.plotter(
        x_full_bCBFQP,
        x_full_vanillaBlending,
        x_full_OI_QP,
        u_act_full_QP,
        u_act_full_vanillaBlending,
        u_act_full_OI_QP,
        u_des_full,
        env2.mu_array,
        env3.mu_array,
        env1,
        phase_plot=True,
        latex_plots=True,
        save_plots=save_plots,
        show_plots=show_plots,
    )
