"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for optimally interpolated safe controllers.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Compares vanilla bCBF-QP, vanilla regulation function bCBF, and our optimal interpolation approach.

"""

from main_sim import Simulation
from plotting_paper_comparison import PaperPlotter
import os
import pickle
from pathlib import Path


if __name__ == "__main__":

    show_plots = False
    save_plots = True
    rerun_sims = False

    base_dir = os.path.dirname(__file__)

    folder = Path(base_dir, "plots")
    folder.mkdir(parents=True, exist_ok=True)

    folder = Path(base_dir, "data")
    folder.mkdir(parents=True, exist_ok=True)

    cache_file = os.path.join(base_dir, "data", "sim_results.pkl")

    if rerun_sims or not os.path.exists(cache_file):
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
            x_full_reg,
            _,
            _,
            u_act_full_reg,
            _,
            _,
            _,
        ) = env2.sim()

        # Run our approach
        print("Running simulation with optimally interpolated QP")
        env3 = Simulation(
            safety_flag=True,
            verbose=False,
            bCBF_QP=False,
            regFun=False,
            OI_QP_flag=True,
        )
        (
            x_full_ours,
            _,
            _,
            u_act_full_ours,
            _,
            _,
            _,
        ) = env3.sim()

        # Save everything
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "x_full_bCBFQP": x_full_bCBFQP,
                    "x_full_reg": x_full_reg,
                    "x_full_ours": x_full_ours,
                    "u_act_full_QP": u_act_full_QP,
                    "u_act_full_reg": u_act_full_reg,
                    "u_act_full_ours": u_act_full_ours,
                    "u_des_full": u_des_full,
                    "phi_bCBF": env1.backupTrajs,
                    "phi_blended": env2.backupTrajs,
                    "phi_OI_QP": env3.backupTrajs,
                    "mu_blended": env2.mu_array,
                    "mu_OI_QP": env3.mu_array,
                },
                f,
            )
    else:
        print("Loading cached results...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)

        x_full_bCBFQP = data["x_full_bCBFQP"]
        x_full_reg = data["x_full_reg"]
        x_full_ours = data["x_full_ours"]
        u_act_full_QP = data["u_act_full_QP"]
        u_act_full_reg = data["u_act_full_reg"]
        u_act_full_ours = data["u_act_full_ours"]
        u_des_full = data["u_des_full"]
        env1 = Simulation(
            safety_flag=True,
            verbose=False,
            bCBF_QP=True,
        )
        env2 = Simulation(
            safety_flag=True,
            verbose=False,
            bCBF_QP=False,
            regFun=True,
            OI_QP_flag=False,
        )
        env3 = Simulation(
            safety_flag=True,
            verbose=False,
            bCBF_QP=False,
            regFun=False,
            OI_QP_flag=True,
        )
        env1.backupTrajs = data["phi_bCBF"]
        env2.backupTrajs = data["phi_blended"]
        env3.backupTrajs = data["phi_OI_QP"]
        env2.mu_array = data["mu_blended"]
        env3.mu_array = data["mu_OI_QP"]

    PaperPlotter.plotter(
        x_full_bCBFQP,
        x_full_reg,
        x_full_ours,
        u_act_full_QP,
        u_act_full_reg,
        u_act_full_ours,
        u_des_full,
        env1,
        env2,
        env3,
        trajectory_plot=True,
        quad_plot=False,
        hexa_plot=True,
        projected_plot=True,
        latex_plots=True,
        save_plots=save_plots,
        show_plots=show_plots,
    )
