import pandas as pd
import numpy as np
from typing import List, Dict
import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import host_subplot
import seaborn as sns
from textwrap import wrap

import data_operations as dtop
import clear_sky as csky
import decorators


describe_with_tukey_fence = decorators.add_tukey_fences(pd.DataFrame.describe)
plt_savefig = decorators.change_savefig(plt.savefig)
legend = decorators.use_default(plt.legend)

#DEV NOTE 13/6/21: TBC which parameters already in thesis
# matplotlibrc locates all the mplstyle located in the same folder (stylelib)
mpl.style.use(['thesis'])
FIGURE_FIGSIZE = tuple(mpl.rcParams['figure.figsize'])
SAVEFIG_FORMAT = mpl.rcParams['savefig.format']

#print(tuple(mpl.rcParams['figure.figsize']))

WRAP_WIDTH = 80
#DEV NOTE 9/6/21 FIGSIZE should be setup and taken from mpl thesis (or whatever is called)
#default value to be adapted for graph; in theory from mpl thesis (or whatever is called)
LGN_BBOX_TO_ANCHOR = decorators._DEFAULT_KWARGS[plt.legend]['bbox_to_anchor']
LGN_LOC =  decorators._DEFAULT_KWARGS[plt.legend]['loc']

DELAY_PARAMETERS_ALL = {'delay_first_h':'sunrise and first',
             'delay_last_h':'sunset and last',
            'delay_transit_h':'sun transit and max',
            'delay_centre_h':'the centre and max'}

MONTHLY_DELAYS_SUPTITLE_ROOT = f'Monthly distribution of time difference between %s measurement'

def plot_cmp_parameter(ax, cmp: pd.DataFrame, column: str, label: str, sl_tz: str):  # tz:str,
    # DEV NOTE 16/5/21 in compare tz is also the index used for df_split
    # DEV NOTE 5/6/21 labelled values to be provided as df and not calculated during plot and retrieved later
    # subplot: violin and swarm need subplots_finalise after subplots end
    # storing median values
    m_values = {}
    m_values["label"] = label
    m_values["column"] = column
    ct = cmp.copy()
    # before splitting to use it later
    ct["month"] = ct.index.to_list()
    ct.month = ct.month.apply(lambda x: x.month)
    # if tz is not None:
    ct_s, ct_w = dtop.df_split_dst(ct, sl_tz=sl_tz, tzi='utc')  # tzi=tz
    # DEV NOTE 26/5/21 average of monthly median instead of median of s/w could be calculated
    md_s = ct_s.loc[:, column].median()
    md_w = ct_w.loc[:, column].median()
    m_values["w"] = md_w
    m_values["s"] = md_s
    # https://dev.to/thalesbruno/subplotting-with-matplotlib-and-seaborn-5ei8
    # -1 since axis positions from 0 due to str labels?
    s_ps = [m - 1 for m in ct_s.month.unique()]
    sns.lineplot(ax=ax, x=s_ps, y=md_s, color='r')
    w_ps = [m - 1 for m in ct_w.month.unique()]
    sns.lineplot(ax=ax, x=[m for m in w_ps if m <= min(s_ps)], y=md_w, color='b')
    sns.lineplot(ax=ax, x=[m for m in w_ps if m >= max(s_ps)], y=md_w, color='b')
    # +0.1 to be on top of line
    ax.text(x=min(w_ps) + 0.5, y=md_w + 0.1, s=f"w md\n{md_w:.2f}", fontsize='large',
            ha='center', va='bottom')
    ax.text(x=min(s_ps) + 0.5, y=md_s + 0.1, s=f"s md\n{md_s:.2f}", fontsize='large',
            ha='center', va='bottom')
    label += f": winter md {md_w:.2f}, summer md {md_s:.2f}"
    sns.violinplot(ax=ax, x='month', y=column, data=ct, inner=None)
    # decreasing size from 5 to 1
    # UserWarning: 60.2% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
    # removing swarm hoping to limit memory issues
    # sns.swarmplot(ax=ax, x='month', y=column, data=ct, color='k', alpha=0.5, size=1)
    # set empty label to avoid taking data label
    ax.set_ylabel("")
    for ty in ax.yaxis.get_major_ticks():
        ty.label.set_fontsize('x-large')
    # creating dataframe
    ymin, ymax = ax.get_ylim()
    m_points = []
    for m in sorted(ct.month.unique()):
        ct_t = ct.loc[ct.month == m, column]
        # tukey not necessary
        dsc = ct_t.describe(percentiles=[0.5])
        md_t = dsc['50%']
        m_values[m] = md_t
        ax.text(x=m - 1, y=ymin, s=f"md {md_t:.2f}", fontsize='large',
                ha='center', va='bottom')
        # not used
        m_points.append(len(ct_t))
    ax.set_xlabel(None)
    # median of months for all to be weigth
    ax.set_title(label, fontsize='xx-large')

    return ax, m_values


def finalise_subplots(fig, ax, fig_title: str, lgn_labels: List[str] = None, xlabel=None, ylabel=None, plt_prm='',
                      bbox_to_anchor=LGN_BBOX_TO_ANCHOR, loc=LGN_LOC):
    # bbox for 9 modified for the bigger legend
    # bbox_to_anchor=(-1.3-0.4, -0.2-0.1) #loc=6
    print(f"Legend: loc {loc}, bbox_to_anchor {bbox_to_anchor}")
    if lgn_labels is not None: ax.legend(labels=lgn_labels, fontsize='xx-large',
                                         bbox_to_anchor=bbox_to_anchor, loc=loc)
    # https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(handles, labels)
    # https://www.kite.com/python/answers/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python
    # Tight layout not applied. The left and right margins cannot be made large enough to accommodate all axes decorations.
    # fig.tight_layout(pad=3.0)
    # https://stackoverflow.com/questions/8248467/matplotlib-tight-layout-doesnt-take-into-account-figure-suptitle
    # Tight layout not applied. The left and right margins cannot be made large enough to accommodate all axes decorations.
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_alpha.html#sphx-glr-gallery-lines-bars-and-markers-fill-between-alpha-py
    fig.suptitle(fig_title, fontsize='xx-large')
    # https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
    # new part from subplots_finalise
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xticks(labels=months, fontsize='xx-large')
    if xlabel is not None: plt.xlabel(xlabel, fontsize='xx-large')
    if ylabel is not None: plt.ylabel(ylabel, fontsize='xx-large')
    fig_title = fig_title.replace('/', '').replace('(', '').replace(')', '').replace(',', '')
    plt_savefig(fname=fig_title + '.' + SAVEFIG_FORMAT)
    #DEV NOTE 9/6/21 remove otherwise not showed
    #plt.close()
    # plt.legend(labels=['gpoa','clear sky model'], fontsize='xx-large', bbox_to_anchor=(1,-0.2), loc=6)
    # fig_filepath = join(decorators._FIG_FOLDER,fig_title+'.jpg')
    # fig_title = fig_title.replace('/','').replace('(','').replace(')','').replace(',','')


def plot_monthly_delays(cmps: List[pd.DataFrame], labels: List[str], sl_tzi: List[str],  # tzi:List[str]=None,
                        parameters: Dict[str, str] = DELAY_PARAMETERS_ALL, timezone_label: str = None,
                        suptitle_root = MONTHLY_DELAYS_SUPTITLE_ROOT):
    # tzi specification not needed since working on dates
    figs_values = pd.DataFrame()

    def plot_parameter(cmps, labels, sl_tzi, cl, pr, timezone_label: str, figs_values: pd.DataFrame):  # tzi,
        fig, axs = plt.subplots(len(cmps), 1, sharex='col', sharey='row') #, figsize=FIGURE_FIGSIZE)
        # sp = 0
        for ax, cmp, label, sl_tz in zip(axs, cmps, labels, sl_tzi):  # , tzi):
            # no ax=ax since AxesSubplot' object is not subscriptable
            ax, ax_values = plot_cmp_parameter(ax=ax, cmp=cmp, column=cl, label=label, sl_tz=sl_tz)  # tz=tz,
            figs_values = figs_values.append(ax_values, ignore_index=True)
            # sp += 1
            # rough solution to avoid warning
            # if sp == len(cmps)-1:
        ax.set_xticklabels(labels=dtop.month_strs, fontsize='x-large')
        suptitle = suptitle_root % pr
        if timezone_label is not None:
            suptitle += f"({timezone_label})"
        finalise_subplots(fig, ax, fig_title=suptitle, xlabel="month",
                          ylabel=f"difference between {pr} measurement [h]")  #
        return figs_values

        # subplots_finalise(fig, suptitle=suptitle,
        #              ylabel=f"difference between {pr} measurement [h]")

    def plot_cmp(cmp, label, sl_tz, parameters, timezone_label: str, figs_values: pd.DataFrame):  # tz,
        fig, axs = plt.subplots(len(parameters), 1, sharex='col', sharey='row') #, figsize=FIGURE_FIGSIZE)
        axs_values = pd.DataFrame()
        sp = 0
        for cl, pr in parameters.items():
            # taking value before iteration
            tt = pr + " measurement datetime difference"
            if len(parameters) > 1:
                axi = axs[sp]
            else:
                axi = axs
            ax, ax_values = plot_cmp_parameter(ax=axi, cmp=cmp, column=cl, label=tt, sl_tz=sl_tz)  # tz=tz,
            figs_values = figs_values.append(ax_values, ignore_index=True)
            sp += 1
            # rough solution to avoid warning
            # if sp == len(parameters.items)-1:
        ax.set_xticklabels(labels=dtop.month_strs, fontsize='x-large')
        suptitle = suptitle_root % f"sun path and {label}" + "s"
        if timezone_label is not None:
            suptitle += f"({timezone_label})"
        finalise_subplots(fig, ax, fig_title=suptitle, xlabel='hour',
                          ylabel=f"difference between sun path and {label} measurements [h]")
        # subplots_finalise(fig, suptitle=suptitle,
        #               ylabel=f"difference between sun path and {label} measurements [h]")
        return figs_values

    if len(cmps) > 1:
        for cl, pr in parameters.items():
            figs_values = plot_parameter(cmps, labels, sl_tzi, cl, pr, timezone_label, figs_values)  # tzi,
    elif len(cmps) == 1:
        figs_values = plot_cmp(cmps[0], labels[0], sl_tzi[0], parameters, timezone_label, figs_values)  # tzi[0],

    return figs_values


def plot_meas_day(mscs, ax, title, columns=None, markers=None):
    def plot_meas(ax, mscs, column=None, marker="k"):
        df = mscs.copy(deep=True)
        df['hour'] = df.index
        df['hour'] = df['hour'].apply(dtop.dt_to_hour)
        # 25/5/21 rough solution to avoid modifying previous code
        if column is None:
            ax.plot(df.loc[:, 'hour'], df.loc[:, 'gpoa'], marker)
            ax.plot(df.loc[:, 'hour'], df.loc[:, 'irradiancetotalpoa'], "b.")
        else:
            ax.plot(df.loc[:, 'hour'], df.loc[:, column], marker)
        return ax
    #if list of
    if isinstance(columns, list) and isinstance(markers, list):
        #if mscs is not a list create it for iteration
        if isinstance(mscs, list) == False:  mscs = [mscs] * len(columns)
        for d, c, m in zip(mscs, columns, markers):
            ax = plot_meas(ax=ax, mscs=d, column=c, marker=m)
    else:
        ax = plot_meas(ax=ax, mscs=mscs, column=None, marker=markers)

    #DEV NOTE 12/6/21 previous versions
    """if isinstance(mscs, pd.DataFrame):
        ax = plot_meas(ax=ax, mscs=mscs, column=None, marker=markers)
    elif isinstance(mscs, list):
        for d, c, m in zip(mscs, columns, markers):
            ax = plot_meas(ax=ax, mscs=d, column=c, marker=m)"""

    # TBC
    # DEV NOTE 15/5/21 limit on number of labels maybe due to size
    for tx in ax.xaxis.get_major_ticks():
        tx.label.set_fontsize('xx-large')
    for ty in ax.yaxis.get_major_ticks():
        ty.label.set_fontsize('xx-large')
    # ax set replacing original ones
    # ax.set_xticklabels(labels=df.hour, fontsize='xx-large')
    # ax.set_yticklabels(labels=range(0,1400,200), fontsize='xx-large')
    # ax.set_ylabel("irradiance W/m2", fontsize='xx-large')
    # ax.set_xlabel("hour", fontsize='xx-large')
    ax.set_title(title, fontsize='xx-large')
    return ax


def plot_daily_measurements(mscss: List[pd.DataFrame], titles: List[str], lgn_labels: List[str], suptitle: str = None,
                            columns: List[str] = None, markers="k.",bbox_to_anchor=LGN_BBOX_TO_ANCHOR, loc=LGN_LOC,
                            plt_prm=''):
    if suptitle is not None:
        n = len(mscss)
        fig_columns = int(round(n ** 0.5, 0))
        fig_rows = int(np.ceil(n / fig_columns))
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_star_poly.html#sphx-glr-gallery-lines-bars-and-markers-scatter-star-poly-py
        fig, axs = plt.subplots(fig_rows, fig_columns, sharex='col', sharey='row', figsize=FIGURE_FIGSIZE)
        # plot counter
        p = 0
    for mscs, title in zip(mscss, titles):
        if suptitle is not None:
            row = int((p) / fig_columns)
            column = int(p % fig_columns)
            if fig_columns > 1:
                ax = axs[row, column]
            else:
                ax = axs[p]
            p += 1
            plot_meas_day(mscs=mscs, ax=ax, title=title, columns=columns, markers=markers)
        else:
            fig, ax = plt.subplots(1, 1, sharex='col', sharey='row') #, figsize=FIGURE_FIGSIZE)
            # no title using supertitle
            plot_meas_day(mscs=mscs, ax=ax, title='', columns=columns, markers=markers)
            finalise_subplots(fig, ax, title+plt_prm, lgn_labels=lgn_labels, plt_prm='',
                              xlabel="utc hour", ylabel="irradiance W/m2", bbox_to_anchor=bbox_to_anchor, loc=loc)
    if suptitle is not None:
        if fig_columns > 1:
            ax = axs[fig_rows - 1, 0]  # using first graph
        else:
            ax = axs[n - 1]
        finalise_subplots(fig, ax, suptitle+plt_prm, lgn_labels=lgn_labels, plt_prm='',
                          xlabel="utc hour", ylabel="irradiance W/m2", bbox_to_anchor=bbox_to_anchor, loc=loc)


def get_date_mscs(df_date: pd.DataFrame(), sl: csky.SolarLibrary, tz: str, shift_h: int):
    df = df_date.copy()
    df['dt'] = df['dt'].apply(lambda x: x + datetime.timedelta(seconds=int(60 * 60 * shift_h)))
    df.index = dtop.reset_tzi_convert_tze(df.loc[:, 'dt'].rename("dt_utc"), tz, 'utc')
    mscs = csky.merge_meas_with_sp(df, sl)
    return mscs


def plot_delay_dates_tzi_shifts(df_c: pd.DataFrame, delays: pd.DataFrame, sl: csky.SolarLibrary, plt_prm=None, label=None,
                                subplots=False):
    # plot parameter for slightly changing the graph during test
    # completeness before since disconnections maybe from one period to another one
    dti = pd.DatetimeIndex(df_c.dt)
    df_c.loc[:, 'date'] = dti.date
    suptitle = 'Comparison of candidate timezones and datetime delays'
    if label is not None:
        suptitle += f" for {label} pyranometer"
    lgn_labels = ['gpoa', 'clear sky model']
    mscss: List[pd.DataFrame] = []
    titles: List[str] = []

    """ 29/5/21 NOT PRIORITY
    dates = delays.d_product_min_date.values
    shift_hs = delays.shift_h.values
    tzs = delays.tz.values   
    titles = delays.apply(lambda x: ("" if subplots else f"{label} pyranometer ")+
                          f'{x["tz"]} {x["shift_h"]}, {x["d"].strftime("%d/%m/%y")}({x["time"][0]})')
    """

    for i, r in delays.iterrows():
        # r["tz","time","shift_h","d_product_min_date","d_product_min_value"]
        tz, time, shift_h, d, prd = r["tz"], r["time"], r["shift_h"], r["d_product_min_date"], r["d_product_min_value"]
        df = df_c.copy(deep=True)
        # selecting by date
        df = df[df.date == d]
        mscs = get_date_mscs(df_date=df, sl=sl, tz=tz, shift_h=shift_h)
        """
        #applying test time shift_h
        df['dt'] = df['dt'].apply(lambda x: x + datetime.timedelta(seconds=int(60*60*shift_h)))
        #converting index before merging. 26/5/21 dt_utc instead of dt_tz
        df.index  = dtop.reset_tzi_convert_tze(df.loc[:,'dt'].rename("dt_utc"), tz, 'utc')
        mscs = csky.merge_meas_with_sp(df, sl)
        """
        # {prd:.2f}
        title = f'{tz} {shift_h}, {d.strftime("%d/%m/%y")}({time[0]})'
        if subplots == False: title = f"{label} pyranometer " + title
        mscss.append(mscs)
        titles.append(title)

    if subplots == False: suptitle = None
    plot_daily_measurements(mscss=mscss, titles=titles, lgn_labels=lgn_labels, suptitle=suptitle)


def plot_meas_sunpath(dfcmp: pd.DataFrame, title="Measurements against sun path", hour_label="utc hour",
                      show_delays=False):
    dfcmp = dtop.add_hour(dfcmp)
    graphs_dfs = []
    graphs_marker = []
    graphs_labels = []
    # sun
    columns_sun = ["sunrise_h", "sunset_h", "transit_h"]
    for c in columns_sun:
        graphs_dfs.append(dfcmp[c])
        graphs_marker.append('k,')
    [graphs_labels.append(i) for i in ['sunrise, sunset and sun transit', None, None]]
    # meas
    [graphs_dfs.append(dfcmp[c]) for c in ["dtfirst_h", "dtmax_h", "dtlast_h"]]
    [graphs_marker.append(m) for m in ['g.', 'y.', 'r.']]
    [graphs_labels.append(l) for l in ["first measurement", "max measurement", "last measurement"]]
    # plot host
    if show_delays:
        host = host_subplot(111)
        par = host.twinx()

    for i in range(0, len(graphs_dfs)):
        if len(graphs_dfs[i]) > 0:
            if show_delays:
                host.plot(graphs_dfs[i].index, graphs_dfs[i], graphs_marker[i], label=graphs_labels[i], alpha=0.5)
            else:
                plt.plot(graphs_dfs[i].index, graphs_dfs[i], graphs_marker[i], label=graphs_labels[i], alpha=0.5)

    if show_delays:
        host.set_ylabel(hour_label)
        host.set_yticks(range(0, 24, 1))
        host.set_xlabel('date')
    else:
        plt.ylabel(hour_label)
        plt.yticks(range(0, 24, 1))
        plt.xlabel('date')
    # reset for parasite
    if show_delays:
        graphs_dfs = [dfcmp[c] for c in ["delay_first_h", "delay_last_h", "delay_transit_h"]]
        graphs_marker = [m for m in ['g^', 'y^', 'r^']]
        graphs_labels = [l for l in ['delay first', 'delay last', 'delay transit']]
        for i in range(0, len(graphs_dfs)):
            if len(graphs_dfs[i]) > 0:
                par.plot(graphs_dfs[i].index, graphs_dfs[i], graphs_marker[i], label=graphs_labels[i], alpha=0.5)
        par.set_ylabel("delay [h]")
    plt.title('\n'.join(wrap(title, WRAP_WIDTH)))

    #plt.legend()  # labels=legend)
    legend()

    plt_savefig = decorators.change_savefig(plt.savefig)
    plt_savefig()


# from typing import List, Dict
# DEV NOTE 10/4/21 clunky function could be optimised with list, tuple and dict

def get_measurement_series(dly: pd.DataFrame, hours_limit, days_limit: int = 24):
    # external to retrieve separetely for first and last
    # @decorators.timer
    dscbig = dly[(dly["after_last_hours"] >= days_limit)]  # .rename({'last_hour':''}, inplace=True)
    rcvbig = dly[(dly["before_first_hours"] >= days_limit)]  # .rename({'last_first':''}, inplace=True)
    dscsmall = dly[(dly["after_last_hours"] > hours_limit) & (dly["after_last_hours"] < days_limit)]
    rcvsmall = dly[(dly["before_first_hours"] > hours_limit) & (
                dly["before_first_hours"] < days_limit)]  # .rename({'last_first':''}, inplace=True)
    first_only = dly[(~dly.index.isin(rcvbig.index)) & (~dly.index.isin(rcvsmall.index))]
    last_only = dly[(~dly.index.isin(dscbig.index)) & (~dly.index.isin(dscsmall.index))]
    return dscbig, rcvbig, dscsmall, rcvsmall, first_only, last_only


@decorators.timer
def plot_completeness(df1_dly: pd.DataFrame, title: str, before_hours_sum: bool = False,
                      suffix1: str = "", df2_dly: pd.DataFrame = None, suffix2: str = "", days_limit=24,
                      annotate1: bool = True, annotate2: bool = False, annotatation_min_hours=48,
                      annotatation_min_hours2=48,
                      ydistmin=100, ydistmin2=100, annotate_verticalshift=5, annotate_horizontalshift=5,
                      hour_label="utc hour"):
    # hour label depend on how data provided, kept convention for normal operations to look regular
    df2notindf1 = []
    bbox_boxstyle = "round"
    bbox_fc = 'w'
    annotate_size = 'xx-small'
    # right to show line after
    annotate_horizontalalignment = 'right'
    annotate_color = 'k'
    annotate_xytext = (0, -1)
    annotate_textcoords = 'offset points'
    annotate_verticalalignment = 'center'
    # shift currently used only for total

    graphs_dfs = []
    graphs_labels = []
    graphs_marker = []
    legend = []

    dly_dsc_1 = describe_with_tukey_fence(df1_dly, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    hours_limit_1 = dly_dsc_1.loc['fence_tukey_max', 'before_first_hours']
    dscbig, rcvbig, dscsmall, rcvsmall, first_only, last_only = get_measurement_series(df1_dly,
                                                                                       hours_limit=hours_limit_1,
                                                                                       days_limit=days_limit)
    dfs_1 = [rcvsmall.first_hour, dscsmall.last_hour, rcvbig.first_hour, dscbig.last_hour,
             first_only.first_hour, last_only.last_hour]
    labels_1 = ['overnight disconnection end', 'overnight disconnection start',
                'day-long disconnection end', 'day-long disconnection start', 'first measurement', 'last measurement']
    markers_1 = ['g^', 'rv', 'g>', 'r<', 'g,', 'r,']

    if 'max_hour' in df1_dly.columns.to_list():
        graphs_dfs.append(df1_dly.max_hour)
        graphs_labels.append('max measurement value' + suffix1)
        graphs_marker.append('y*')

    for i in range(len(dfs_1)):
        if len(dfs_1[i]) > 0:
            # test
            # print(len(dfs_1[i]), labels_1[i]+suffix1, markers_1[i])
            graphs_dfs.append(dfs_1[i])
            graphs_labels.append(labels_1[i] + suffix1)
            graphs_marker.append(markers_1[i])

    if df2_dly is not None:
        dly_dsc_2 = describe_with_tukey_fence(df2_dly, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        hours_limit_2 = dly_dsc_2.loc['fence_tukey_max', 'before_first_hours']
        dscbig2, rcvbig2, dscsmall2, rcvsmall2, first_only2, last_only2 = get_measurement_series(df2_dly,
                                                                                                 hours_limit=hours_limit_2,
                                                                                                 days_limit=days_limit)

        dfs_2 = [rcvsmall2.first_hour, dscsmall2.last_hour, rcvbig2.first_hour, dscbig2.last_hour,
                 first_only2.first_hour, last_only2.last_hour]

        markers_2 = ['b^', 'mv', 'b>', 'm<', 'b,', 'm,']

        for i in range(len(dfs_2)):
            # for 2o visualising only series differences between the two dataframes
            dff = dtop.series_differences(dfs_1[i], dfs_2[i])
            df2notindf1.append(dff)
            if len(dff) > 0:
                graphs_dfs.append(dff)
                graphs_labels.append(labels_1[i] + suffix2)
                graphs_marker.append(markers_2[i])

    if all(s in df1_dly.columns.to_list() for s in ['sunrise_h', 'sunset_h', 'transit_h']):
        [graphs_dfs.append(i) for i in [df1_dly.sunrise_h, df1_dly.sunset_h, df1_dly.transit_h]]
        [graphs_marker.append(i) for i in ['k,', 'k,', 'k,']]
        [graphs_labels.append(i) for i in ['sunrise, sunset and sun transit', None, None]]

    host = host_subplot(111)
    par = host.twinx()

    for i in range(0, len(graphs_dfs)):
        if len(graphs_dfs[i]) > 0:
            # alpha=0.5 for transparency
            host.plot(graphs_dfs[i].index, graphs_dfs[i], graphs_marker[i], label=graphs_labels[i], alpha=0.5)
            # legend.append(graphs_labels[i])

    if before_hours_sum == True:
        _ = lambda x: x - hours_limit_1 if x > hours_limit_1 else 0
        df1_dly["before_first_hours_cml"] = pd.Series(
            [df1_dly.loc[:x, 'before_first_hours'].apply(_).sum() for x in df1_dly.index],
            index=df1_dly.index)
        label = "cumulative disconnection hours"
        par.set_ylabel(label)
        p2, = par.plot(df1_dly.index, df1_dly.before_first_hours_cml, ":", label=label + suffix1)

        # 21/4/21
        # legend.append(label+suffix1)

        # DEV NOTE 24/04/21 fixed ydistmin but dynamic from figsize & y could be used ?
        def hrz_alg(r, annotate_horizontalalignment, before_first_hours_cml, ydist=ydistmin):
            if r["before_first_hours_cml"] - before_first_hours_cml < ydist:
                if annotate_horizontalalignment == 'right':
                    annotate_horizontalalignment = 'left'
                    horizontalshift = 1
                # elif annotate_horizontalalignment =='left':
                else:
                    annotate_horizontalalignment = 'right'
                    horizontalshift = -1
            else:
                annotate_horizontalalignment = 'right'
                horizontalshift = 0
            return annotate_horizontalalignment, horizontalshift * annotate_horizontalshift

        bbox = dict(boxstyle=bbox_boxstyle, fc=bbox_fc, edgecolor="blue")
        ann_str = f'tot{str(suffix1)}: {max(df1_dly.before_first_hours_cml) / 24:.0f}d, {max(df1_dly.before_first_hours_cml) / dtop.tdhour(max(df1_dly.index) - min(df1_dly.index)) * 100:.2f}%'
        par.annotate(ann_str,
                     (max(df1_dly.index), max(df1_dly.before_first_hours_cml)),
                     bbox=bbox, size=annotate_size, horizontalalignment="right",
                     xytext=(annotate_xytext[0], annotate_xytext[1] + annotate_verticalshift),
                     textcoords=annotate_textcoords,
                     verticalalignment=annotate_verticalalignment)

        before_first_hours_cml = 0

        if annotate1:
            long_dsc1 = df1_dly.loc[df1_dly.before_first_hours > annotatation_min_hours,
                                    ["before_first_hours", "first_datetime", "before_first_hours_cml"]]
            for i, r in long_dsc1.iterrows():
                bfhl = f'{r["before_first_hours"] / 24:.0f}d' if r[
                                                                     "before_first_hours"] > 24 else f'{r["before_first_hours"]:.0f}h'
                ann_str = bfhl + f', {r["first_datetime"].strftime("%m/%d %H:%M")}'
                annotate_horizontalalignment, hs = hrz_alg(r, annotate_horizontalalignment, before_first_hours_cml,
                                                           ydistmin)
                par.annotate(ann_str,
                             (i, r["before_first_hours_cml"]),
                             bbox=bbox, size=annotate_size, horizontalalignment=annotate_horizontalalignment,
                             xytext=(annotate_xytext[0] + hs, annotate_xytext[1]), textcoords=annotate_textcoords,
                             verticalalignment=annotate_verticalalignment)

                before_first_hours_cml = r["before_first_hours_cml"]

        if df2_dly is not None:
            df2_dly["before_first_hours_cml"] = pd.Series(
                [df2_dly.loc[:x, 'before_first_hours'].apply(_).sum() for x in df2_dly.index],
                index=df2_dly.index)
            label = "cumulative disconnection hours"
            par.set_ylabel(label)
            p2, = par.plot(df2_dly.index, df2_dly.before_first_hours_cml, ":", label=label + suffix2)
            # legend.append()

            # 22/4/22
            bbox = dict(boxstyle=bbox_boxstyle, fc=bbox_fc, edgecolor='orange')
            ann_str = f'tot{str(suffix2)}: {max(df2_dly.before_first_hours_cml) / 24:.0f}d, {max(df2_dly.before_first_hours_cml) / dtop.tdhour(max(df2_dly.index) - min(df2_dly.index)) * 100:.2f}%'
            par.annotate(ann_str,
                         (max(df2_dly.index), max(df2_dly.before_first_hours_cml)),
                         bbox=bbox, size=annotate_size, horizontalalignment="left",
                         xytext=(
                         annotate_xytext[0] + annotate_horizontalshift, annotate_xytext[1] + annotate_verticalshift),
                         textcoords=annotate_textcoords,
                         verticalalignment=annotate_verticalalignment)

            before_first_hours_cml = 0
            if annotate2:
                # filtering minimum only for main not for differences
                if annotate1:
                    bfh_dff = dtop.series_differences(df1_dly.before_first_hours, df2_dly.before_first_hours)
                    day_dsc2 = df2_dly.loc[bfh_dff.index,
                                           ["before_first_hours", "first_datetime", "before_first_hours_cml"]]
                # elif annotate1 == False:
                else:
                    day_dsc2 = df2_dly[["before_first_hours", "first_datetime", "before_first_hours_cml"]]
                day_dsc2 = day_dsc2.loc[day_dsc2.before_first_hours > annotatation_min_hours2,
                                        ["before_first_hours", "first_datetime", "before_first_hours_cml"]]
                for i, r in day_dsc2.iterrows():
                    bfhl = f'{r["before_first_hours"] / 24:.0f}d' if r[
                                                                         "before_first_hours"] > 24 else f'{r["before_first_hours"]:.0f}h'
                    ann_str = bfhl + f', {r["first_datetime"].strftime("%m/%d %H:%M")}'
                    annotate_horizontalalignment, hs = hrz_alg(r, annotate_horizontalalignment, before_first_hours_cml,
                                                               ydistmin2)

                    par.annotate(ann_str,
                                 (i, r["before_first_hours_cml"]),
                                 bbox=bbox, size=annotate_size, horizontalalignment=annotate_horizontalalignment,
                                 xytext=(annotate_xytext[0] + hs, annotate_xytext[1]), textcoords=annotate_textcoords,
                                 verticalalignment=annotate_verticalalignment)

                    before_first_hours_cml = r["before_first_hours_cml"]


    #plt.legend()  # labels=legend)
    legend()

    plt.title('\n'.join(wrap(title, WRAP_WIDTH)))
    host.set_ylabel(hour_label)
    host.set_yticks(range(0, 24, 1))
    # This method should only be used after fixing the tick positions using Axes.set_yticks.
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yticklabels.html?highlight=set_yticklabels#matplotlib.axes.Axes.set_yticklabels
    # host.set_yticklabel()
    host.set_xlabel('date')
    # plt.show() not required
    plt_savefig = decorators.change_savefig(plt.savefig)
    plt_savefig()

    return df2notindf1