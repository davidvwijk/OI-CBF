"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for optimally interpolated safe controllers.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module contains plotting functions.

"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.ticker as ticker
import pikepdf


class PaperPlotter:
    def plotter(
        x_bCBFQP,
        x_reg,
        x_ours,
        u_act_bCBFQP,
        u_act_reg,
        u_act_ours,
        u_des,
        env,
        env2,
        env3,
        trajectory_plot=True,
        quad_plot=False,
        hexa_plot=True,
        projected_plot=True,
        latex_plots=False,
        save_plots=False,
        show_plots=True,
    ):
        if latex_plots:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                }
            )
            plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

        def plot_vertical_plane(
            ax, ng, pg, width=10, height=13.5, color="royalblue", alpha=0.2
        ):
            """
            Plot a vertical geofence plane in 3D defined by a 2D normal and a reference point.
            """
            ng = np.array(ng, dtype=float)
            pg = np.array(pg, dtype=float)

            nN, nE = ng
            pN_star, pE_star = pg
            d = nN * pN_star + nE * pE_star
            pN_vals = np.array([-width, width])
            H_vals = np.array([9, height])

            if abs(nE) > 1e-8:
                PE1 = (d - nN * pN_vals[0]) / nE
                PE2 = (d - nN * pN_vals[1]) / nE
                verts = [
                    [
                        [pN_vals[0], PE1, H_vals[0]],
                        [pN_vals[0], PE1, H_vals[1]],
                        [pN_vals[1], PE2, H_vals[1]],
                        [pN_vals[1], PE2, H_vals[0]],
                    ]
                ]
            else:
                # Vertical plane aligned with pN
                PE_vals = np.array([-width, width]) + pE_star - 10
                verts = [
                    [
                        [pN_star, PE_vals[0], H_vals[0]],
                        [pN_star, PE_vals[0], H_vals[1]],
                        [pN_star, PE_vals[1], H_vals[1]],
                        [pN_star, PE_vals[1], H_vals[0]],
                    ]
                ]

            plane = Poly3DCollection(
                verts,
                alpha=alpha,
                facecolor=color,
                zorder=1,
            )
            ax.add_collection3d(plane)

            edge_lines = [
                [verts[0][0], verts[0][1]],  # left vertical
                [verts[0][1], verts[0][2]],  # top
                [verts[0][2], verts[0][3]],  # right vertical
                [verts[0][3], verts[0][0]],  # bottom
            ]

            edge_collection = Line3DCollection(
                edge_lines,
                colors="blue",
                linewidths=3,
                alpha=1.0,  # full opacity for edges
            )
            ax.add_collection3d(edge_collection)

            # Label
            mid = np.mean(verts[0], axis=0)
            ax.text3D(
                mid[0],
                mid[1] * 0.75,
                mid[2] * 1.1,
                "geofence",
                zdir="y",
                rotation=45,
                color="blue",
                fontsize=legend_sz,
            )

        def format_sig_figs(value, pos):
            return f"{value:.1g}"  # '3g' ensures 3 significant figures

        # Convert from m to km
        x_bCBFQP[3:6] /= 1000
        x_reg[3:6] /= 1000
        x_ours[3:6] /= 1000
        env.geo_pg /= 1000

        # Styling parameters
        axis_sz, legend_sz, ticks_sz = 25, 20, 19
        lwp = 3
        border_lw = 1.7

        # Define colors
        flow_color = "#bababa"
        bCBF_color = "black"
        blend_color = "#D82796"
        blendQP_color = "#27D869"
        u_des_color = "#DB0000"

        # Arrays
        delta_t = env.del_t
        t_span_u = np.arange(u_act_reg.shape[1] - 1) * delta_t
        t_span = np.arange(x_bCBFQP.shape[1]) * delta_t

        # 3D trajectory plot
        if trajectory_plot:
            fig = plt.figure(figsize=(9, 9))
            ax = fig.add_subplot(111, projection="3d")

            # Plot geofence
            plot_vertical_plane(ax, env.geo_ng, env.geo_pg)

            # Plot trajectories
            ax.plot(
                x_bCBFQP[3, :],
                x_bCBFQP[4, :],
                x_bCBFQP[5, :],
                color=bCBF_color,
                linewidth=lwp,
                label="bCBF",
                zorder=99,
            )

            ax.plot(
                x_reg[3, :],
                x_reg[4, :],
                x_reg[5, :],
                color=blend_color,
                linewidth=lwp,
                label="Blended",
                zorder=99,
            )

            ax.plot(
                x_ours[3, :],
                x_ours[4, :],
                x_ours[5, :],
                color=blendQP_color,
                linewidth=lwp,
                label="Ours",
                zorder=99,
            )

            # Start marker
            ax.scatter(
                x_bCBFQP[3, 0],
                x_bCBFQP[4, 0],
                x_bCBFQP[5, 0],
                color="black",
                s=50,
                marker="*",
                zorder=5,
            )

            step_freq = 5
            if env3.backupTrajs:
                max_numBackup = len(env3.backupTrajs)
                for i, xy in enumerate(env3.backupTrajs[::step_freq]):
                    if i == 0:
                        label = r"$\boldsymbol{\phi}_{\rm b}(\tau,\boldsymbol{x})" + "$"
                    else:
                        label = None
                    if i < max_numBackup:
                        ax.plot(
                            xy[:, 3] / 1000,
                            xy[:, 4] / 1000,
                            xy[:, 5] / 1000,
                            "--",
                            color=flow_color,
                            linewidth=lwp * 0.8,
                            zorder=1,
                            label=label,
                        )

            # Labels
            ax.set_xlabel("$p_{\\rm N}$ (km)", fontsize=axis_sz, labelpad=6)
            ax.set_ylabel("$p_{\\rm E}$ (km)", fontsize=axis_sz, labelpad=6)
            ax.set_zlabel("H (km)", fontsize=axis_sz, labelpad=6)

            ax.tick_params(axis="x", pad=-1)
            ax.tick_params(axis="y", pad=1)

            # Styling
            ax.tick_params(axis="both", labelsize=ticks_sz)

            ax.legend(
                fontsize=legend_sz,
                loc="center",
                bbox_to_anchor=(0.25, 0.67),
                fancybox=True,
                shadow=True,
                handletextpad=0.3,
                handlelength=1.5,
            )

            ax.grid(False)
            ax.view_init(21, -116)
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

            if save_plots:
                plt.savefig(
                    "plots/trajectory_plot.pdf",
                    pad_inches=0,
                )

                pdf = pikepdf.open("plots/trajectory_plot.pdf")
                page = pdf.pages[0]
                mediabox = page.mediabox

                mediabox[0] = float(mediabox[0]) + 5  # left
                mediabox[1] = float(mediabox[1]) + 40  # bottom
                mediabox[2] = float(mediabox[2]) - 15  # right
                mediabox[3] = float(mediabox[3]) - 75  # top

                pdf.save("plots/trajectory_plot_cropped.pdf")

        if projected_plot:
            axes_ar = []
            fig = plt.figure(figsize=(3, 9), dpi=100)

            ax = fig.add_subplot(111)
            axes_ar.append(ax)

            y_max = 25.1
            y_min = 24.4
            x_max = 0.1
            x_min = -17

            mul_fnt = 1.2

            unsafe_color = "blue"
            gray_background = "#eeeded"
            unsafe_background = "#979797"
            lwp_sets = 2.5
            alpha_set = 0.4

            ax.set_xlim([x_max, x_min])
            ax.set_ylim([y_min, y_max])
            plt.xticks(fontsize=ticks_sz * mul_fnt)
            plt.yticks(fontsize=ticks_sz * mul_fnt)
            plt.xlabel("$p_{\\rm E}$ (km)", fontsize=axis_sz * mul_fnt)
            plt.ylabel("$p_{\\rm N}$ (km)", fontsize=axis_sz * mul_fnt)

            ax.plot(
                x_bCBFQP[4, :],
                x_bCBFQP[3, :],
                color=bCBF_color,
                linewidth=lwp,
            )

            ax.plot(
                x_reg[4, :],
                x_reg[3, :],
                color=blend_color,
                linewidth=lwp,
            )

            ax.plot(
                x_ours[4, :],
                x_ours[3, :],
                color=blendQP_color,
                linewidth=lwp,
            )

            plt.fill_between(
                [x_min, x_max],
                y_min,
                env.geo_pg[0],
                color=gray_background,
                alpha=alpha_set,
            )

            plt.fill_between(
                [x_min, x_max],
                env.geo_pg[0],
                y_max,
                color=unsafe_background,
                alpha=alpha_set,
            )
            plt.hlines(
                xmin=x_min,
                xmax=x_max,
                y=env.geo_pg[0],
                color=unsafe_color,
                linewidth=lwp_sets,
                linestyles="-",
            )

            for a in axes_ar:
                for spine in a.spines.values():
                    spine.set_linewidth(border_lw)

            ax.text(
                -5.5,
                25.07,
                "unsafe",
                fontsize=legend_sz * mul_fnt,
                color="#585858",
                zorder=99,
            )

            ax.text(
                -8.5,
                25.013,
                "geofence",
                fontsize=legend_sz * mul_fnt,
                color=unsafe_color,
                zorder=99,
            )

            if save_plots:
                plt.savefig(
                    "plots/proj_aircraft.pdf",
                    bbox_inches="tight",
                    pad_inches=0,
                )

        # Legend stuff
        bp = 0.15  # Border pad
        frame_bool = True
        fancy_bool = True
        txtpd = 0.4
        lwp_quad = 2.0
        lwp_inset = 1.8

        # Styling parameters
        axis_sz, legend_sz, ticks_sz = 16, 14, 14
        border_lw = 1

        ticks_sz_quad = ticks_sz * 0.8
        axis_sz_quad = axis_sz * 0.8

        if quad_plot:
            axes_ar = []

            fig = plt.figure(figsize=(8, 3.5), dpi=100)

            ################### u1 plot ###################
            ax = fig.add_subplot(2, 2, 1)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz_quad)
            plt.yticks(fontsize=ticks_sz_quad)
            ax.set_xlim([0, t_span[-1]])

            ax.set_ylabel("$u_1 \,{\\rm (rad/s)}$", fontsize=axis_sz_quad)

            ax.plot(
                t_span_u,
                u_act_bCBFQP[0, 1:],
                "-",
                color=bCBF_color,
                label="bCBF",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_reg[0, 1:],
                "-",
                color=blend_color,
                label="Blended",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_ours[0, 1:],
                "-",
                color=blendQP_color,
                label="Ours",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_des[0, 1:],
                "--",
                color=u_des_color,
                label="Nominal",
                linewidth=lwp_quad,
            )
            ax.legend(
                fontsize=legend_sz * 0.8,
                loc="center",
                bbox_to_anchor=(1.04, 1.22),
                columnspacing=0.6,
                handletextpad=0.2,
                handlelength=1.5,
                fancybox=fancy_bool,
                frameon=frame_bool,
                shadow=False,
                framealpha=1,
                facecolor="white",
                ncol=4,
            )
            ax.yaxis.set_major_formatter(FuncFormatter(format_sig_figs))
            ax.set_xticklabels([])
            ##################### u2 plot #####################

            ax = fig.add_subplot(2, 2, 3)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz_quad)
            plt.yticks(fontsize=ticks_sz_quad)
            ax.set_xlim([0, t_span[-1]])
            ax.set_ylabel("$u_2$", fontsize=axis_sz_quad)

            ax.plot(
                t_span_u,
                u_act_bCBFQP[1, 1:],
                "-",
                color=bCBF_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_reg[1, 1:],
                "-",
                color=blend_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_ours[1, 1:],
                "-",
                color=blendQP_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_des[1, 1:],
                "--",
                color=u_des_color,
                linewidth=lwp_quad,
            )
            plt.xlabel("time (s)", fontsize=axis_sz_quad)

            ##################### mu plot #####################
            ax = fig.add_subplot(2, 2, 2)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz_quad)
            plt.yticks(fontsize=ticks_sz_quad)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span_u,
                env2.mu_array,
                "-",
                color=blend_color,
                label=r"$\mu$",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                env3.mu_array,
                "-",
                color=blendQP_color,
                label=r"$\mu^{\star}$",
                linewidth=lwp_quad,
            )
            ax.legend(
                fontsize=legend_sz,
                loc="lower right",
                bbox_to_anchor=(1.02, -0.05),
                frameon=True,
                borderpad=bp,
                columnspacing=0.3,
                fancybox=True,
                shadow=False,
                framealpha=0.93,
                facecolor="white",
                handlelength=1.1,
                handletextpad=txtpd,
                ncol=3,
            )
            ax.set_xticklabels([])

            ##################### h(x) plot ######################
            ax = fig.add_subplot(2, 2, 4)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz_quad)
            plt.yticks(fontsize=ticks_sz_quad)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span,
                env.h_x_geo_array(x_bCBFQP) * 1000,
                "-",
                color=bCBF_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span,
                env.h_x_geo_array(x_reg) * 1000,
                "-",
                color=blend_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span,
                env.h_x_geo_array(x_ours) * 1000,
                "-",
                color=blendQP_color,
                linewidth=lwp_quad,
            )
            plt.axhline(
                0,
                color="k",
                linestyle="--",
                linewidth=lwp_quad,
            )
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.xlabel("time (s)", fontsize=axis_sz_quad)

            # Create zoomed-in inset
            axins = inset_axes(
                ax,
                width="65%",
                height="50%",
                loc="upper right",
            )
            axins.plot(
                t_span,
                env.h_x_geo_array(x_bCBFQP) * 1000,
                "-",
                color=bCBF_color,
                linewidth=lwp_inset,
            )
            axins.plot(
                t_span,
                env.h_x_geo_array(x_reg) * 1000,
                "-",
                color=blend_color,
                linewidth=lwp_inset,
            )
            axins.plot(
                t_span,
                env.h_x_geo_array(x_ours) * 1000,
                "-",
                color=blendQP_color,
                linewidth=lwp_inset,
            )
            axins.axhline(
                0,
                color="k",
                linestyle="--",
                linewidth=lwp_inset,
            )
            x1, x2 = 40, t_span[-1]
            y1, y2 = -100, 250
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            plt.xticks(fontsize=ticks_sz_quad * 0.5)
            plt.yticks(fontsize=ticks_sz_quad * 0.5)
            conn_lines = mark_inset(
                ax, axins, loc1=3, loc2=4, fc="none", ec="black", lw=1, ls="-", zorder=0
            )
            ax.set_ylabel(r"$h(\boldsymbol{x})$", fontsize=axis_sz_quad)
            for a in axes_ar:
                for spine in a.spines.values():
                    spine.set_linewidth(border_lw)

            if save_plots:
                plt.savefig(
                    "plots/quad_plot_aircraft.pdf",
                    bbox_inches="tight",
                    pad_inches=0,
                )

        if hexa_plot:
            label_size = 12
            axes_ar = []
            fig = plt.figure(figsize=(8, 3.5 * (3 / 2)), dpi=100)
            ax = fig.add_subplot(3, 2, 1)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz_quad)
            plt.yticks(fontsize=ticks_sz_quad)
            ax.set_xlim([0, t_span[-1]])
            ax.set_ylabel("$u_1 \,{\\rm (rad/s)}$", fontsize=axis_sz_quad)
            ax.plot(
                t_span_u,
                u_act_bCBFQP[0, 1:],
                "-",
                color=bCBF_color,
                label="bCBF",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_reg[0, 1:],
                "-",
                color=blend_color,
                label="Blended",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_ours[0, 1:],
                "-",
                color=blendQP_color,
                label="Ours",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_des[0, 1:],
                "--",
                color=u_des_color,
                label="Nominal",
                linewidth=lwp_quad,
            )
            ax.legend(
                fontsize=legend_sz * 0.8,
                loc="center",
                bbox_to_anchor=(1.04, 1.22),
                columnspacing=0.6,
                handletextpad=0.2,
                handlelength=1.5,
                fancybox=fancy_bool,
                frameon=frame_bool,
                shadow=False,
                framealpha=1,
                facecolor="white",
                ncol=4,
            )
            ax.yaxis.set_major_formatter(FuncFormatter(format_sig_figs))
            ax.set_xticklabels([])
            ax.text(
                0.02,
                0.95,
                "(a)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=label_size,
                color="black",
            )
            ax = fig.add_subplot(3, 2, 3)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz_quad)
            plt.yticks(fontsize=ticks_sz_quad)
            ax.set_xlim([0, t_span[-1]])
            ax.set_ylabel("$u_2$", fontsize=axis_sz_quad)

            ax.plot(
                t_span_u,
                u_act_bCBFQP[1, 1:],
                "-",
                color=bCBF_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_reg[1, 1:],
                "-",
                color=blend_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_ours[1, 1:],
                "-",
                color=blendQP_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_des[1, 1:],
                "--",
                color=u_des_color,
                linewidth=lwp_quad,
            )
            ax.set_xticklabels([])
            ax.text(
                0.02,
                0.95,
                "(c)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=label_size,
                color="black",
            )

            ##################### mu plot #####################
            ax = fig.add_subplot(3, 2, 2)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz_quad)
            plt.yticks(fontsize=ticks_sz_quad)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span_u,
                env2.mu_array,
                "-",
                color=blend_color,
                label=r"$\mu$",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                env3.mu_array,
                "-",
                color=blendQP_color,
                label=r"$\mu^{\star}$",
                linewidth=lwp_quad,
            )
            ax.legend(
                fontsize=legend_sz,
                loc="lower right",
                bbox_to_anchor=(1.02, -0.05),
                frameon=True,
                borderpad=bp,
                columnspacing=0.3,
                fancybox=True,
                shadow=False,
                framealpha=0.93,
                facecolor="white",
                handlelength=1.1,
                handletextpad=txtpd,
                ncol=3,
            )
            ax.set_xticklabels([])

            ax.text(
                0.02,
                0.95,
                "(b)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=label_size,
                color="black",
            )

            ax = fig.add_subplot(3, 2, 4)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz_quad)
            plt.yticks(fontsize=ticks_sz_quad)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span,
                env.h_x_geo_array(x_bCBFQP) * 1000,
                "-",
                color=bCBF_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span,
                env.h_x_geo_array(x_reg) * 1000,
                "-",
                color=blend_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span,
                env.h_x_geo_array(x_ours) * 1000,
                "-",
                color=blendQP_color,
                linewidth=lwp_quad,
            )
            plt.axhline(
                0,
                color="k",
                linestyle="--",
                linewidth=lwp_quad,
            )
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axins = inset_axes(
                ax,
                width="65%",
                height="50%",
                loc="upper right",
            )
            axins.plot(
                t_span,
                env.h_x_geo_array(x_bCBFQP) * 1000,
                "-",
                color=bCBF_color,
                linewidth=lwp_inset,
            )
            axins.plot(
                t_span,
                env.h_x_geo_array(x_reg) * 1000,
                "-",
                color=blend_color,
                linewidth=lwp_inset,
            )
            axins.plot(
                t_span,
                env.h_x_geo_array(x_ours) * 1000,
                "-",
                color=blendQP_color,
                linewidth=lwp_inset,
            )
            axins.axhline(
                0,
                color="k",
                linestyle="--",
                linewidth=lwp_inset,
            )
            x1, x2 = 40, t_span[-1]
            y1, y2 = -100, 250
            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)

            plt.xticks(fontsize=ticks_sz_quad * 0.5)
            plt.yticks(fontsize=ticks_sz_quad * 0.5)
            conn_lines = mark_inset(
                ax, axins, loc1=3, loc2=4, fc="none", ec="black", lw=1, ls="-", zorder=0
            )
            ax.set_ylabel(r"$h(\boldsymbol{x})$", fontsize=axis_sz_quad)
            ax.text(
                0.02,
                0.235,
                "(d)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=label_size,
                color="black",
            )

            ax.set_xticklabels([])

            # x = [phi, theta, psi, pN, pE, H, P, Nz]
            #### Roll
            ax = fig.add_subplot(3, 2, 5)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz_quad)
            plt.yticks(fontsize=ticks_sz_quad)
            ax.set_xlim([0, t_span[-1]])
            plt.xlabel("time (s)", fontsize=axis_sz_quad)
            plt.ylabel(r"$\phi \, ({\rm rad})$", fontsize=axis_sz_quad, labelpad=-2.0)

            ax.plot(
                t_span,
                x_bCBFQP[0, :],
                "-",
                color=bCBF_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span,
                x_reg[0, :],
                "-",
                color=blend_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span,
                x_ours[0, :],
                "-",
                color=blendQP_color,
                linewidth=lwp_quad,
            )
            ax.text(
                0.02,
                0.95,
                "(e)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=label_size,
                color="black",
            )

            # x = [phi, theta, psi, pN, pE, H, P, Nz]
            #### Pitch
            ax = fig.add_subplot(3, 2, 6)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz_quad)
            plt.yticks(fontsize=ticks_sz_quad)
            ax.set_xlim([0, t_span[-1]])
            plt.xlabel("time (s)", fontsize=axis_sz_quad)
            plt.ylabel(r"$\theta \, ({\rm rad})$", fontsize=axis_sz_quad)

            ax.plot(
                t_span,
                x_bCBFQP[1, :],
                "-",
                color=bCBF_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span,
                x_reg[1, :],
                "-",
                color=blend_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span,
                x_ours[1, :],
                "-",
                color=blendQP_color,
                linewidth=lwp_quad,
            )

            ax.text(
                0.02,
                0.95,
                "(f)",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=label_size,
                color="black",
            )

            for a in axes_ar:
                for spine in a.spines.values():
                    spine.set_linewidth(border_lw)

            if save_plots:
                plt.savefig(
                    "plots/hexa_plot_aircraft.pdf",
                    dpi=100,
                    bbox_inches="tight",
                    pad_inches=0,
                )

        if show_plots:
            plt.show()
