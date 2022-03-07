#!/usr/bin/env python3
# https://github.com/biolab/orange3/blob/3.15.0/Orange/evaluation/scoring.py

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import sys
from pandas import read_excel, DataFrame
from scipy.stats import friedmanchisquare
from matplotlib.backends.backend_agg import FigureCanvasAgg

from merge import new_path, scoring_path


def compute_CD(avranks, n, alpha="0.05", test="nemenyi"):
    """
    Returns critical difference for Nemenyi or Bonferroni-Dunn test
    according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested datasets N. Test can be either "nemenyi" for
    for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.
    """
    k = len(avranks)
    d = {("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                               2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                               3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                               3.426041, 3.458425, 3.488685, 3.517073,
                               3.543799, 3.569, 3.593, 3.616, 3.637, 3.658,
                               3.678, 3.696, 3.714, 3.732, 3.749,
                               3.765, 3.780, 3.795, 3.810, 3.824,
                               3.837, 3.850, 3.863, 3.876, 3.888,
                               3.899, 3.911, 3.922, 3.933, 3.943,
                               3.954, 3.964, 3.973, 3.983, 3.992],
         ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                              2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                              2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                              3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
         ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                       2.638, 2.690, 2.724, 2.773],
         ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                      2.394, 2.450, 2.498, 2.539]}
    q = d[(test, alpha)]
    cd = q[k] * (k * (k + 1) / (6.0 * n)) ** 0.5
    return cd


def graph_ranks(avranks, names, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
    """
    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    tempsort = sorted([(a, i) for i, a in enumerate(sums)], reverse=reverse)
    ssums = nth(tempsort, 0)
    sortidx = nth(tempsort, 1)
    nnames = [names[x] for x in sortidx]

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    if cd and cdmethod is None:
        # add scale
        distanceh = 0.25
        cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=0.7)

    bigtick = 0.1
    smalltick = 0.05

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=0.7)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom")

    k = len(ssums)
    m, M = min(ssums), max(ssums)

    def color(x: float):
        if M - m < cd:
            pass
        elif M - m < 2 * cd:
            return 'r' if 2 * x < m + M else 'g'
        elif x < m + cd:
            return 'r'
        elif x > M - cd:
            return 'g'
        return 'b'

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)], color(ssums[i]),
             linewidth=0.7)
        text(textspace - 0.2, chei, nnames[i], ha="right", va="center")

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * 0.2
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)], color(ssums[i]),
             linewidth=0.7)
        text(textspace + scalewidth + 0.2, chei, nnames[i],
             ha="left", va="center")

    if cd and cdmethod is None:
        # upper scale
        if not reverse:
            begin, end = rankpos(lowv), rankpos(lowv + cd)
        else:
            begin, end = rankpos(highv), rankpos(highv - cd)

        line([(begin, distanceh), (end, distanceh)], linewidth=0.7)
        line([(begin, distanceh + bigtick / 2),
              (begin, distanceh - bigtick / 2)],
             linewidth=0.7)
        line([(end, distanceh + bigtick / 2),
              (end, distanceh - bigtick / 2)],
             linewidth=0.7)

        cd_msg = "CD = %.3f" % cd
        p_value = kwargs.pop('p_value')
        p_value_msg = "Friedman p-value: %.3e" % p_value
        text((begin + end) / 2, distanceh - 0.05, cd_msg, ha="center", va="bottom")
        text(rankpos(highv), distanceh, p_value_msg, ha="right", va="bottom")

    if filename:
        print_figure(fig, filename, **kwargs)
    return fig, [(names[i], color(sums[i])) for i in range(k)]


def rank(df: DataFrame):
    for row in df.values:
        row[row.argsort()] = np.arange(len(row), 0, -1)
    avg_ranks = df.mean()
    cd = compute_CD(avg_ranks, len(df))
    stat, p_value = friedmanchisquare(*df.values.T)
    # return graph_ranks(avg_ranks, df.columns, cd, p_value=p_value)
    return graph_ranks(avg_ranks, df.columns, cd, width=10, p_value=p_value)


def main(paths: str):
    # 如果paths是一文件
    if not os.path.isdir(paths):
        if os.path.splitext(paths)[1] == ".xlsx":
            df = read_excel(paths, index_col=0).astype(float)
            fig, mess = rank(df)
            mess_str = '\n'.join(','.join(pair[i] for pair in mess) for i in range(2))
            with open(new_path(paths, '.csv'), 'w') as f:
                print(mess_str, file=f)
            fig.canvas.set_window_title(paths)
            canvas = FigureCanvasAgg(fig)
            for ext in ['.png', '.ps', '.eps', '.pdf']:
                canvas.print_figure(scoring_path(paths, ext))
    # 如果paths是文件夹
    else:
        print("正在处理文件夹内文件")
        for path in os.listdir(paths):
            if os.path.splitext(path)[1] == ".xlsx":
                path = os.path.join(paths, path)
                df = read_excel(path, index_col=0).astype(float)
                fig, mess = rank(df)
                mess_str = '\n'.join(','.join(pair[i] for pair in mess) for i in range(2))
                with open(scoring_path(path, '.csv'), 'w') as f:
                    print(mess_str, file=f)
                fig.canvas.set_window_title(path)
                canvas = FigureCanvasAgg(fig)
                for ext in ['.png', '.ps', '.eps', '.pdf']:
                    canvas.print_figure(scoring_path(path, ext))
        print("文件处理完毕")


if __name__ == '__main__':
    plt.rc('font', family='Times New Roman', weight='bold')
    paths = sys.argv[1]
    main(paths)
