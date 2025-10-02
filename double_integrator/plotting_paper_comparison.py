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
from matplotlib.ticker import FuncFormatter


class PaperPlotter:
    def plotter(
        x_bCBFQP,
        x_vanillaBlending,
        x_full_blendQP,
        u_act_bCBFQP,
        u_act_vanillaBlending,
        u_act_full_blendQP,
        u_des,
        mu_array,
        mu_star_array,
        env,
        phase_plot=True,
        quad_plot=False,
        bi_plot=True,
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

        alpha_set = 0.4
        gray_background = "#eeeded"
        xaxis_sz, legend_sz, ticks_sz = 27, 23, 25
        lblsize = legend_sz * 1.35
        lwp_sets = 2.5
        x1 = x_bCBFQP[0, :]
        x2 = x_bCBFQP[1, :]

        x1 = x_bCBFQP[0, :]
        x2 = x_bCBFQP[1, :]

        def setupPhasePlot():
            # Plot limits
            x_max = 0.35
            x_min = -2.3
            y_max = 1.05
            y_min = -0.3
            x_c = np.linspace(x_min, 0, 1000)
            plt.figure(figsize=(12.5, 8.5), dpi=100)

            plt.fill_between(
                [0, x_max],
                y_min,
                y_max,
                color="#A6A5A5",
                alpha=alpha_set,
            )
            plt.vlines(
                x=0.0,
                ymin=y_min,
                ymax=y_max,
                color="#b8b7b7",
                linewidth=lwp_sets,
                linestyles="-",
            )
            plt.hlines(
                y=0.0,
                xmin=x_min,
                xmax=0,
                # color="#bdbdbd",
                color="#a3a3a3",
                linewidth=lwp_sets,
                linestyles="-",
                alpha=1,
            )
            plt.fill_between(
                x_c,
                0,
                y_min,
                color="#d4d4d4",
                alpha=alpha_set,
            )
            plt.fill_between(
                [x_min, 0],
                0,
                y_max,
                color=gray_background,
                alpha=alpha_set,
            )
            plt.text(-0.74, -0.2, "$\mathcal{C}_{\\rm B}$", fontsize=lblsize)
            plt.text(
                -0.33,
                0.9,
                "$\mathcal{C}_{\\rm S} \\backslash \mathcal{C}_{\\rm B}$",
                fontsize=lblsize,
            )
            plt.text(
                0.044,
                0.8,
                "$\mathcal{X} \\backslash \mathcal{C}_{\\rm S}$",
                fontsize=lblsize,
            )

            plt.axis("equal")
            plt.xlabel(r"$x_1$", fontsize=xaxis_sz, labelpad=0)
            plt.ylabel(r"$x_2$", fontsize=xaxis_sz, labelpad=0)
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")

        ##################################################################################
        ##################################################################################

        # Styling
        lwp = 2.8
        border_lw = 1.5

        # Legend stuff
        bp = 0.25  # Border pad
        frame_bool = True
        shadow_bool = False
        fancy_bool = True
        txtpd = 0.4

        lw_backup = 2.4

        # Colors
        blend_color = "#D82796"
        blendQP_color = "#27D869"
        bCBF_color = "black"
        mu_color = blend_color
        mu_star_color = blendQP_color
        flow_color = "#bababa"
        color_primary = "#DB0000"

        if phase_plot:
            setupPhasePlot()
            ax = plt.gca()
            ax.plot(
                x1[0],
                x2[0],
                "k*",
                markersize=9,
                label=None,
                zorder=800,
            )

            plt.plot(
                x_vanillaBlending[0, :],
                x_vanillaBlending[1, :],
                "-",
                color=blend_color,
                linewidth=lwp,
                label="Blended",
            )
            arrow_indices = np.arange(8, len(x_vanillaBlending[0, :]), 20)
            for i in arrow_indices:
                ax.annotate(
                    "",
                    xy=(x_vanillaBlending[0, :][i + 1], x_vanillaBlending[1, :][i + 1]),
                    xytext=(x_vanillaBlending[0, :][i], x_vanillaBlending[1, :][i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=blend_color,
                        lw=lwp * 1.1,
                    ),
                )

            plt.plot(
                x_full_blendQP[0, :],
                x_full_blendQP[1, :],
                linestyle=(0, (5, 4)),
                color=blendQP_color,
                linewidth=lwp,
                label="Ours",
                zorder=9999,
            )
            arrow_indices = np.arange(8, len(x_full_blendQP[0, :]), 20)
            for i in arrow_indices:
                ax.annotate(
                    "",
                    xy=(x_full_blendQP[0, :][i + 1], x_full_blendQP[1, :][i + 1]),
                    xytext=(x_full_blendQP[0, :][i], x_full_blendQP[1, :][i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=blendQP_color,
                        lw=lwp * 1.2,
                    ),
                    zorder=9998,
                )

            label = "bCBF"
            plt.plot(
                x1, x2, "-", color=bCBF_color, linewidth=lwp, label=label, zorder=999
            )
            arrow_indices = np.arange(8, len(x1), 20)
            for i in arrow_indices:
                ax.annotate(
                    "",
                    xy=(x1[i + 1], x2[i + 1]),
                    xytext=(x1[i], x2[i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=bCBF_color,
                        lw=lwp * 1.1,
                    ),
                    zorder=800,
                )

            if env.backupTrajs:
                max_numBackup = 17
                for i, xy in enumerate(env.backupTrajs):
                    if i == 0:
                        label = r"$\boldsymbol{\phi}_{\rm b}(\tau,\boldsymbol{x})" + "$"
                    else:
                        label = None

                    if i < max_numBackup:
                        plt.plot(
                            xy[:, 0],
                            xy[:, 1],
                            "--",
                            color=flow_color,
                            linewidth=lw_backup,
                            label=label,
                            zorder=0,
                        )

            for spine in ax.spines.values():
                spine.set_linewidth(border_lw)

            ax.legend(
                fontsize=legend_sz * 0.94,
                loc="lower left",
                framealpha=1,
                ncol=1,
                columnspacing=0.4,
                handletextpad=0.2,
                handlelength=1.6,
                fancybox=fancy_bool,
                frameon=frame_bool,
                borderpad=bp,
                shadow=shadow_bool,
            )

            if save_plots:
                plt.savefig(
                    "plots/comparison_phase_plot_full.pdf",
                    bbox_inches="tight",
                    pad_inches=0,
                )

        xaxis_sz, legend_sz, ticks_sz = 23, 20, 22

        def format_sig_figs(value, pos):
            return f"{value:.1g}"  # '3g' ensures 3 significant figures

        delta_t = env.del_t
        t_span_u = np.arange(u_act_bCBFQP.shape[1] - 1) * delta_t
        t_span = np.arange(len(x1)) * delta_t

        # Legend stuff
        bp = 0.14  # Border pad
        frame_bool = True
        shadow_bool = False
        fancy_bool = True
        txtpd = 0.4
        lwp_quad = 2.4

        axes_ar = []
        border_lw = 1.2

        if bi_plot:
            fig = plt.figure(figsize=(10, 4.5 / 2), dpi=100)

            ax = fig.add_subplot(1, 2, 1)
            axes_ar.append(ax)
            ax.set_xlim([0, t_span[-1]])  # Adjust the x-axis limits
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)

            ax.plot(
                t_span_u,
                mu_array[:],
                "-",
                color=mu_color,
                label="$\mu$",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                mu_star_array[:],
                "-",
                color=mu_star_color,
                label="$\mu^{\\star}$",
                linewidth=lwp_quad,
            )
            ax.legend(
                fontsize=legend_sz,
                loc="lower right",
                columnspacing=0.3,
                bbox_to_anchor=(1.02, -0.05),
                handletextpad=txtpd,
                handlelength=1.3,
                fancybox=fancy_bool,
                frameon=frame_bool,
                shadow=shadow_bool,
                borderpad=bp,
                ncol=4,
            )
            ax.yaxis.set_major_formatter(FuncFormatter(format_sig_figs))
            plt.xlabel("time (s)", fontsize=xaxis_sz)

            ax = fig.add_subplot(1, 2, 2)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span_u,
                u_des[0][1:],
                "--",
                color=color_primary,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_bCBFQP[0][1:],
                "-",
                color=bCBF_color,
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_vanillaBlending[0][1:],
                "-",
                color=blend_color,
                label="$k_{\\rm m}$",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act_full_blendQP[0][1:],
                linestyle=(0, (5, 3)),
                color=blendQP_color,
                label="$k^{\\star}_{\\rm m}$",
                linewidth=lwp_quad,
            )
            ax.text(
                5.4,
                0.69,
                "$k_{\\rm p} = 1$",
                fontsize=legend_sz,
                color=color_primary,
            )
            ax.legend(
                fontsize=legend_sz,
                loc="lower right",
                bbox_to_anchor=(1.02, -0.05),
                frameon=frame_bool,
                borderpad=bp,
                columnspacing=0.3,
                fancybox=fancy_bool,
                shadow=shadow_bool,
                handlelength=1.1,
                handletextpad=txtpd,
                ncol=3,
            )
            plt.xlabel("time (s)", fontsize=xaxis_sz)
            plt.subplots_adjust(wspace=0.21)
            for a in axes_ar:
                for spine in a.spines.values():
                    spine.set_linewidth(border_lw)

            if save_plots:
                plt.savefig(
                    "plots/bi_plot_full.pdf",
                    bbox_inches="tight",
                    pad_inches=0,
                )

        if show_plots:
            plt.show()
