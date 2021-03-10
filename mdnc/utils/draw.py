#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Utilities - Extended visualization tools (mpl)
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   numpy 1.13+
#   matplotlib 3.1.1+
# Extended figure drawing tools. This module is based on
# matplotlib and provides some fast interfacies for drawing
# some specialized figures (like loss function).
################################################################
'''

import itertools
import contextlib

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = ['setFigure', 'use_tex', 'fix_log_axis',
           'AxisMultipleTicker',
           'plot_hist', 'plot_bar', 'plot_scatter', 'plot_training_records', 'plot_error_curves', 'plot_distribution_curves']


class setFigure(contextlib.ContextDecorator):
    '''setFigure context decorator.
    A context decorator class, which is used for changing the figure's
    configurations locally for a specific function.
    An example is
    ```python
    @mdnc.utils.draw.setFigure(font_size=12)
    def plot_curve():
        ...
    ```
    Could be also used as
    ```python
    with mdnc.utils.draw.setFigure(font_size=12):
        ...
    ```
    Arguments:
        style: the style of the figure. The available list could be
               referred here:
               https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        font_name: the local font family name for the output figure.
        font_size: the local font size for the output figure.
        use_tex: whether to use LaTeX backend for the output figure.
    '''
    def __init__(self, style=None, font_name=None, font_size=None, use_tex=None):
        self.style = style
        self.font_name = font_name
        self.font_size = font_size
        self.use_tex = use_tex

        self.__stacks = dict()

    @contextlib.contextmanager
    def __set_font(self):
        '''Context for setting font.'''
        restore = dict()
        if self.font_name:
            restore['font.family'] = mpl.rcParams['font.family']
            restore['font.sans-serif'] = mpl.rcParams['font.sans-serif']
            mpl.rcParams['font.family'] = 'sans-serif'
            mpl.rcParams['font.sans-serif'] = [self.font_name, ]
        if self.font_size:
            restore['font.size'] = mpl.rcParams['font.size']
            mpl.rcParams['font.size'] = self.font_size
        try:
            yield
        finally:
            for k, v in restore.items():
                mpl.rcParams[k] = v

    def __set_style(self):
        '''Context for adding style configurations.'''
        if self.style:
            return plt.style.context(self.style)
        else:
            return contextlib.nullcontext()

    @contextlib.contextmanager
    def __set_tex(self):
        '''Context for adding LaTeX configurations.'''
        if self.use_tex is not None:
            restore = dict()
            useafm = mpl.rcParams.get('ps.useafm', None)
            if useafm is not None:
                restore['ps.useafm'] = useafm
            use14corefonts = mpl.rcParams.get('pdf.use14corefonts', None)
            if use14corefonts is not None:
                restore['pdf.use14corefonts'] = use14corefonts
            usetex = mpl.rcParams.get('text.usetex', None)
            if usetex is not None:
                restore['text.usetex'] = usetex
            mpl.rcParams['ps.useafm'] = self.use_tex
            mpl.rcParams['pdf.use14corefonts'] = self.use_tex
            mpl.rcParams['text.usetex'] = self.use_tex
        try:
            yield
        finally:
            if self.use_tex is not None:
                for k, v in restore.items():
                    mpl.rcParams[k] = v

    def __enter__(self):
        self.__stacks.clear()
        self.__stacks['style'] = self.__set_style()
        self.__stacks['font'] = self.__set_font()
        self.__stacks['tex'] = self.__set_tex()
        self.__stacks['style'].__enter__()
        self.__stacks['font'].__enter__()
        self.__stacks['tex'].__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__stacks['tex'].__exit__(exc_type, exc_value, exc_traceback)
        self.__stacks['font'].__exit__(exc_type, exc_value, exc_traceback)
        self.__stacks['style'].__exit__(exc_type, exc_value, exc_traceback)
        self.__stacks.clear()


def use_tex(flag=False):
    '''Switch the maplotlib backend to LaTeX.
    This function is not recommended. Please use
    `mdnc.utils.draw.setFigure` as a safer way.
    Arguments:
        flag: a bool, indicating whether to use the LaTeX backend
            for rendering figure fonts.
    '''
    mpl.rcParams['ps.useafm'] = flag
    mpl.rcParams['pdf.use14corefonts'] = flag
    mpl.rcParams['text.usetex'] = flag


def fix_log_axis(ax=None, axis='y'):
    '''Control the log axis to be limited in 10^n ticks.
    Arguments:
        ax: the axis that requires to be controlled. If set None,
            the plt.gca() would be used.
        axis: x, y or 'xy'.
    '''
    if ax is None:
        ax = plt.gca()
    if axis.find('y') != -1:
        ymin, ymax = np.log10(ax.get_ylim())
        ymin = np.floor(ymin) if ymin - np.floor(ymin) < 0.3 else ymin
        ymax = np.ceil(ymax) if np.ceil(ymax) - ymax < 0.3 else ymax
        ax.set_ylim(*np.power(10.0, np.asarray([ymin, ymax])).tolist())
    if axis.find('x') != -1:
        xmin, xmax = np.log10(ax.get_xlim())
        xmin = np.floor(xmin) if xmin - np.floor(xmin) < 0.3 else xmin
        xmax = np.ceil(xmax) if np.ceil(xmax) - xmax < 0.3 else xmax
        ax.set_xlim(*np.power(10.0, np.asarray([xmin, xmax])).tolist())


class AxisMultipleTicker:
    '''Use multiple locator to define the formatted axis.
    Inspired by the following post:
        https://stackoverflow.com/a/53586826
    '''

    def __init__(self, den_major=2, den_minor=5, number=np.pi, symbol=r'\pi'):
        '''Initialization.
        Arguments:
            den_major: the denominator for the major ticks.
            den_minor: the denominator for the minor ticks.
            number:    the value that each "symbol" represents.
            symbol:    the displayed symbol of the major ticks.
        '''
        self.den_major = int(den_major)
        if self.den_major <= 0:
            raise ValueError('utils.draw: The argument "den_major" requires to be >0.')
        self.den_minor = int(den_minor)
        if self.den_minor <= 1:
            raise ValueError('utils.draw: The argument "den_minor" requires to be >1.')
        self.number = number
        if self.number <= 0.0:
            raise ValueError('utils.draw: The argument "number" requires to be >0.')
        self.symbol = symbol
        if not isinstance(symbol, str):
            raise TypeError('utils.draw: The argument "symbol" requires to be an str.')

    @property
    def major_locator(self):
        '''Return the major locator. Use axis.set_major_locator() to set it.'''
        return plt.MultipleLocator(self.number / self.den_major)

    @property
    def minor_locator(self):
        '''Return the minor locator. Use axis.set_minor_locator() to set it.'''
        return plt.MultipleLocator(self.number / (self.den_minor * self.den_major))

    @property
    def formatter(self):
        '''Return the major formatter. Use axis.set_major_formatter() to set it.'''
        return plt.FuncFormatter(self.__multiple_formatter())

    def __multiple_formatter(self):
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        denominator = self.den_major
        number = self.number
        symbol = self.symbol

        def _multiple_formatter(x, pos):
            den = denominator
            num = np.int(np.rint(den * x / number))
            com = gcd(num, den)
            num, den = int(num / com), int(den / com)
            if den == 1:
                if num == 0:
                    return r'$0$'
                if num == 1:
                    return r'${0}$'.format(symbol)
                elif num == -1:
                    return r'$-{0}$'.format(symbol)
                else:
                    return r'${0}{1}$'.format(num, symbol)
            else:
                if num == 1:
                    return r'$\frac{{ {0} }}{{ {1} }}$'.format(symbol, den)
                elif num == -1:
                    return r'$-\frac{{ {0} }}{{ {1} }}$'.format(symbol, den)
                else:
                    sign = '-' if num < 0 else ''
                    return r'${0}\frac{{ {1}{2} }}{{ {3} }}$'.format(sign, num, symbol, den)
        return _multiple_formatter


def __plot_configs(xlabel=None, ylabel=None, figure_size=None,
                   has_legend=False, legend_loc=None, legend_col=None,
                   fig=None, ax=None):
    '''Final configurations shared by all utilities.'''
    fig = fig if fig is not None else plt.gcf()
    ax = ax if ax is not None else plt.gca()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if figure_size:
        fig.set_size_inches(*figure_size)
    plt.tight_layout(rect=[0.03, 0, 0.97, 1])
    if has_legend:
        kwargs = dict()
        if legend_loc is not None:
            kwargs['loc'] = legend_loc
        if legend_col is not None:
            kwargs['ncol'] = legend_col
        ax.legend(labelspacing=0., **kwargs)


def plot_hist(gen, normalized=False, cumulative=False,
              xlabel='Value', ylabel='Number of samples',
              x_log=False, y_log=False,
              figure_size=(6, 5.5), legend_loc=None, legend_col=None,
              fig=None, ax=None):
    '''Plot a histogram for multiple distributions.
    Arguments:
        gen: a generator callable object (function), each "yield"
             returns a sample. It allows users to provide an extra
             kwargs dict for each iteration. For each iteration it
             returns 1 1D data.
        normalized: whether to use normalization for each group
                    when drawing the histogram.
        xlabel: the x axis label.
        ylabel: the y axis label.
        x_log: whether to convert the x axis into the log repre-
               sentation.
        y_log: whether to convert the y axis into the log repre-
               sentation.
        figure_size: the size of the output figure.
        legend_loc: the localtion of the legend. (The legend only
                    works when passing `label` to each iteration)
        legend_col: the column of the legend.
        fig: a figure instance. If not given, would use plt.gcf()
             for instead.
        ax: a subplot instance. If not given, would use plt.gca()
             for instead.
    '''
    fig = fig if fig is not None else plt.gcf()
    ax = ax if ax is not None else plt.gca()
    # Get iterator
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    # Set scale
    if x_log:
        plt.xscale('log')
    # Begin to parse data
    has_legend = False
    for data in gen:
        c = next(cit)
        if isinstance(data, (tuple, list)) and len(data) > 1 and isinstance(data[-1], dict):
            kwargs = data[-1]
            data = data[0]
        else:
            kwargs = dict()
        has_legend = 'label' in kwargs
        kwargs.update(c)
        ax.hist(data, alpha=0.8, density=normalized, cumulative=cumulative, log=y_log, **kwargs)
    __plot_configs(xlabel=xlabel, ylabel=ylabel, figure_size=figure_size,
                   has_legend=has_legend, legend_loc=legend_loc, legend_col=legend_col,
                   fig=fig, ax=ax)


def plot_bar(gen, num,
             xlabel=None, ylabel='value',
             x_tick_labels=None, y_log=False,
             figure_size=(6, 5.5), legend_loc=None, legend_col=None,
             fig=None, ax=None):
    '''Plot a bar graph for multiple result groups.
    Arguments:
        gen: a generator callable object (function), each "yield"
             returns a sample. It allows users to provide an extra
             kwargs dict for each iteration. For each iteration it
             returns 1 1D data.
        num: the total number of data samples thrown by the
             generator.
        xlabel: the x axis label.
        ylabel: the y axis label.
        x_tick_labels: the x tick labels that is used for
                       overriding the original value [0, 1, 2,
                       ...].
        y_log: whether to convert the y axis into the log repre-
               sentation.
        figure_size: the size of the output figure.
        legend_loc: the localtion of the legend. (The legend only
                    works when passing `label` to each iteration)
        legend_col: the column of the legend.
        fig: a figure instance. If not given, would use plt.gcf()
             for instead.
        ax: a subplot instance. If not given, would use plt.gca()
             for instead.
    '''
    fig = fig if fig is not None else plt.gcf()
    ax = ax if ax is not None else plt.gca()
    # Get iterators
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    width = 0.75
    width = tuple(zip(np.linspace(-width / 2, width / 2, num + 1)[:-1], width / num * np.ones(num)))
    wit = itertools.cycle(width)
    # Get tick labels
    if x_tick_labels is not None:
        x_tick_labels = list(x_tick_labels)
        x = np.arange(len(x_tick_labels))
    else:
        x = None
    # Set scale
    if y_log:
        ax.set_yscale('log')
    # Begin to parse data
    has_legend = False
    for data in gen:
        c = next(cit)
        wp, w = next(wit)
        if isinstance(data, (tuple, list)) and len(data) > 1 and isinstance(data[-1], dict):
            kwargs = data[-1]
            data = data[0]
        else:
            kwargs = dict()
        has_legend = 'label' in kwargs
        kwargs.update(c)
        if x is None:
            x = np.arange(len(data))
        ax.bar(x + wp + w / 2, data, w, **kwargs)
    if x_tick_labels is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(x_tick_labels)
    __plot_configs(xlabel=xlabel, ylabel=ylabel, figure_size=figure_size,
                   has_legend=has_legend, legend_loc=legend_loc, legend_col=legend_col,
                   fig=fig, ax=ax)


def plot_scatter(gen,
                 xlabel=None, ylabel='value',
                 x_log=None, y_log=False,
                 figure_size=(6, 5.5), legend_loc=None, legend_col=None,
                 fig=None, ax=None):
    '''Plot a scatter graph for multiple data groups.
    Arguments:
        gen: a generator callable object (function), each "yield"
             returns a sample. It allows users to provide an extra
             kwargs dict for each iteration. For each iteration,
             it returns 2 1D arrays or a 2D array.
        xlabel: the x axis label.
        ylabel: the y axis label.
        x_log: whether to convert the x axis into the log repre-
               sentation.
        y_log: whether to convert the y axis into the log repre-
               sentation.
        figure_size: the size of the output figure.
        legend_loc: the localtion of the legend. (The legend only
                    works when passing `label` to each iteration)
        legend_col: the column of the legend.
        fig: a figure instance. If not given, would use plt.gcf()
             for instead.
        ax: a subplot instance. If not given, would use plt.gca()
             for instead.
    '''
    fig = fig if fig is not None else plt.gcf()
    ax = ax if ax is not None else plt.gca()
    # Get iterators
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    mit = itertools.cycle(['o', '^', 's', 'd', '*', 'P'])
    # Set scale
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')
    # Begin to parse data
    has_legend = False
    for data in gen:
        c, m = next(cit), next(mit)
        if isinstance(data, (tuple, list)) and len(data) > 1 and isinstance(data[-1], dict):
            kwargs = data[-1]
            if len(data) == 3:
                l_m, l_d = data[:2]
            elif data[0].shape[0] == 2:
                l_m, l_d = data[0]
            else:
                raise ValueError('utils.draw: Input data should be two 1D arrays or '
                                 'one 2D array with a shape of [2, L]')
        else:
            kwargs = dict()
        has_legend = 'label' in kwargs
        kwargs.update(c)
        ax.scatter(l_m, l_d, marker=m, **kwargs)
    __plot_configs(xlabel=xlabel, ylabel=ylabel, figure_size=figure_size,
                   has_legend=has_legend, legend_loc=legend_loc, legend_col=legend_col,
                   fig=fig, ax=ax)


def plot_training_records(gen,
                          xlabel=None, ylabel='value',
                          x_mark_num=None, y_log=False,
                          figure_size=(6, 5.5), legend_loc=None, legend_col=None,
                          fig=None, ax=None):
    '''Plot a training curve graph for multiple data groups.
    Arguments:
        gen: a generator callable object (function), each "yield"
             returns a sample. It allows users to provide an extra
             kwargs dict for each iteration. For each iteration,
             it returns 4 1D arrays, or 2 2D arrays, or 2 1D arrays,
             or a 4D array, or a 2D array, or a 1D array.
        xlabel: the x axis label.
        ylabel: the y axis label.
        x_mark_num: the number of markers for the x axis.
        y_log: whether to convert the y axis into the log repre-
               sentation.
        figure_size: the size of the output figure.
        legend_loc: the localtion of the legend. (The legend only
                    works when passing `label` to each iteration)
        legend_col: the column of the legend.
        fig: a figure instance. If not given, would use plt.gcf()
             for instead.
        ax: a subplot instance. If not given, would use plt.gca()
             for instead.
    '''
    fig = fig if fig is not None else plt.gcf()
    ax = ax if ax is not None else plt.gca()
    # Get iterators
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    mit = itertools.cycle(['o', '^', 's', 'd', '*', 'P'])
    # Set scale
    if y_log:
        ax.set_yscale('log')
    # Begin to parse data
    kwargs = dict()
    has_legend = False
    for data in gen:
        c, m = next(cit), next(mit)
        has_valid = None
        if isinstance(data, (tuple, list)):
            if isinstance(data[-1], dict):
                *data, kwargs = data
            if len(data) == 4:  # 4 1D data tuple.
                x, v, val_x, val_v = data
                has_valid = True
            elif len(data) == 2:  # 2 data tuple.
                d1, d2 = data
                if d1.ndim == 2 and d2.ndim == 2 and d1.shape[0] == 2 and d2.shape[0] == 2:
                    # 2 2D data.
                    x, v = d1
                    val_x, val_v = d2
                    has_valid = True
                elif d1.ndim == 1 and d2.ndim == 1:
                    # 2 1D data.
                    x, v = d1, d2
                    has_valid = False
                else:
                    raise ValueError('utils.draw: The input data shape is invalid, when using'
                                     'data sequence, there should be 4 1D data, or'
                                     ' 2 2D data, or 2 1D data.')
            elif len(data) == 1:
                data = data[0]
        if has_valid is None:
            if data.ndim == 2:
                if len(data) == 4:
                    x, v, val_x, val_v = data
                    has_valid = True
                elif len(data) == 2:
                    x, v = data
                    has_valid = False
                else:
                    raise ValueError('utils.draw: The input data shape is invalid, when using'
                                     'a single array, it should be 4D data, or'
                                     ' 2D data, or 1D data.')
            elif data.ndim == 1:
                x = np.arange(0, len(data))
                v = data
                has_valid = False
            else:
                raise ValueError('utils.draw: The input data shape is invalid, when using'
                                 'a single array, it should be 4D data, or'
                                 ' 2D data, or 1D data.')
        has_legend = 'label' in kwargs and kwargs['label'] is not None
        if 'color' not in kwargs:
            kwargs.update(c)
        base_label = kwargs.pop('label', None)
        get_label = base_label
        if isinstance(base_label, (list, tuple)):
            get_label = base_label[0]
        elif (base_label is not None) and has_valid:
            get_label = base_label + ' (train)'
        if x_mark_num is not None:
            x_mark = np.round(np.linspace(0, len(x) - 1, x_mark_num)).astype(np.int)
        else:
            x_mark = None
        marker = m if x_mark is not None else None
        ms = 7 if x_mark is not None else None
        ax.plot(x, v, marker=marker, ms=ms, label=get_label, markevery=x_mark, **kwargs)
        if has_valid:
            if isinstance(base_label, (list, tuple)):
                get_label = base_label[1]
            elif base_label is not None:
                get_label = base_label + ' (valid)'
            if x_mark_num is not None:
                x_mark = np.round(np.linspace(0, len(val_x) - 1, x_mark_num)).astype(np.int)
            else:
                x_mark = None
            marker = m if x_mark is not None else None
            ms = 7 if x_mark is not None else None
            ax.plot(val_x, val_v, linestyle='--', marker=marker, ms=ms, markevery=x_mark, label=get_label, **kwargs)
    __plot_configs(xlabel=xlabel, ylabel=ylabel, figure_size=figure_size,
                   has_legend=has_legend, legend_loc=legend_loc, legend_col=legend_col,
                   fig=fig, ax=ax)


def plot_error_curves(gen, x_error_num=10,
                      y_error_method='std', plot_method='error',
                      xlabel=None, ylabel='value',
                      y_log=False,
                      figure_size=(6, 5.5), legend_loc=None, legend_col=None,
                      fig=None, ax=None):
    '''Plot lines with error bars for multiple data groups.
    Arguments:
        gen: a generator callable object (function), each "yield"
             returns a sample. It allows users to provide an extra
             kwargs dict for each iteration. For each iteration,
             it returns 1D + 2D arrays, or a single 2D array.
        x_error_num: the number of displayed error bars.
        y_error_method: the method for calculating the error bar.
            (1) std: use standard error.
            (2) minmax: use the range of the data.
        plot_method: the method for plotting the figure.
            (1) error: use error bar graph.
            (2) fill:  use fill_between graph.
        xlabel: the x axis label.
        ylabel: the y axis label.
        y_log: whether to convert the y axis into the log repre-
               sentation.
        figure_size: the size of the output figure.
        legend_loc: the localtion of the legend. (The legend only
                    works when passing `label` to each iteration)
        legend_col: the column of the legend.
        fig: a figure instance. If not given, would use plt.gcf()
             for instead.
        ax: a subplot instance. If not given, would use plt.gca()
             for instead.
    '''
    fig = fig if fig is not None else plt.gcf()
    ax = ax if ax is not None else plt.gca()
    # Get iterators
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    mit = itertools.cycle(['o', '^', 's', 'd', '*', 'P'])
    # Set scale
    if y_log:
        ax.set_yscale('log')
    # Begin to parse data
    has_legend = False
    kwargs = dict()
    for data in gen:
        c, m = next(cit), next(mit)
        x = None
        if isinstance(data, (tuple, list)):
            if isinstance(data[-1], dict):
                *data, kwargs = data
            if len(data) == 2:  # 2 1D data tuple.
                x, data = data
            elif len(data) == 1:
                data = data[0]
            else:
                raise ValueError('utils.draw: The input data list is invalid, it should'
                                 'contain 1D + 2D array or a 2D array.')
        if data.ndim == 2:
            if x is None:
                x = np.arange(0, len(data))
            v = data
        else:
            raise ValueError('utils.draw: The input data list is invalid, it should'
                             'contain 1D + 2D array or a 2D array.')
        has_legend = 'label' in kwargs
        kwargs.update(c)
        avg = np.mean(v, axis=1)
        if y_error_method == 'minmax':
            geterr = np.stack((avg - np.amin(v, axis=1), np.amax(v, axis=1) - avg), axis=0)
        else:
            geterr = np.repeat(np.expand_dims(np.std(v, axis=1), axis=0), 2, axis=0)
        if plot_method == 'fill':
            mark_every = np.round(np.linspace(0, len(x) - 1, x_error_num)).astype(np.int).tolist()
            ax.plot(x, avg, marker=m, ms=7, markevery=mark_every, **kwargs)
            ax.fill_between(x, avg - geterr[0, ...], avg + geterr[1, ...], alpha=0.3, color=c['color'])
        else:
            error_every = len(x) // x_error_num
            ax.errorbar(x, avg, yerr=geterr, errorevery=error_every, marker=m, ms=5, markevery=error_every, **kwargs)
    __plot_configs(xlabel=xlabel, ylabel=ylabel, figure_size=figure_size,
                   has_legend=has_legend, legend_loc=legend_loc, legend_col=legend_col,
                   fig=fig, ax=ax)


def plot_distribution_curves(gen, method='mean', level=3, outlier=0.1,
                             xlabel=None, ylabel='value',
                             y_log=False,
                             figure_size=(6, 5.5), legend_loc=None, legend_col=None,
                             fig=None, ax=None):
    '''Plot lines with multi-level distribution for multiple data groups.
    This function has similar meaning of plot_error_curves. It is
    used for compressing the time-series histograms. Its output is
    similar to tensorboard.distribution.
    Arguments:
        gen: a sample generator, each "yield" returns a sample. It
             allows users to provide an extra kwargs dict for each
             iteration. For each iteration it returns 1D + 2D arrays,
             or a single 2D array.
        method: the method for calculating curves, use 'mean' or
                'middle'.
        level: the histogram level.
        outlier: outlier proportion, the part marked as outliers
                 would be thrown away when drawing the figures.
        xlabel: the x axis label.
        ylabel: the y axis label.
        y_log: whether to convert the y axis into the log repre-
               sentation.
        figure_size: the size of the output figure.
        legend_loc: the localtion of the legend. (The legend only
                    works when passing `label` to each iteration)
        legend_col: the column of the legend.
        fig: a figure instance. If not given, would use plt.gcf()
             for instead.
        ax: a subplot instance. If not given, would use plt.gca()
             for instead.
    '''
    fig = fig if fig is not None else plt.gcf()
    ax = ax if ax is not None else plt.gca()
    if level < 1:
        raise TypeError('utils.draw: Histogram level should be at least 1.')
    if method not in ('mean', 'middle'):
        raise TypeError('utils.draw: The curve calculation method should be either "mean" or "middle".')
    # Get iterators
    cit = itertools.cycle(mpl.rcParams['axes.prop_cycle'])
    mit = itertools.cycle(['o', '^', 's', 'd', '*', 'P'])
    # Set scale
    if y_log:
        ax.set_yscale('log')
    # Begin to parse data
    has_legend = False
    kwargs = dict()
    for data in gen:
        c, m = next(cit), next(mit)
        x = None
        if isinstance(data, (tuple, list)):
            if isinstance(data[-1], dict):
                *data, kwargs = data
            if len(data) == 2:  # 4 1D data tuple.
                x, data = data
            elif len(data) == 1:
                data = data[0]
            else:
                raise ValueError('utils.draw: The input data list is invalid, it should'
                                 'contain 1D + 2D array or a 2D array.')
        if data.ndim == 2:
            if x is None:
                x = np.arange(0, len(data))
            v = data
        else:
            raise ValueError('utils.draw: The input data list is invalid, it should'
                             'contain 1D + 2D array or a 2D array.')
        has_legend = 'label' in kwargs
        kwargs.update(c)
        vsort = np.sort(v, axis=1)
        N = v.shape[1]
        if method == 'middle':
            avg = vsort[:, N // 2]
        else:
            avg = np.mean(v, axis=1)
        # Calculate ranges according to levels:
        vu, vd = [], []
        for i in range(level):
            if method == 'middle':
                pos = max(1, int(np.round((outlier + (1 - outlier) * ((i - 1) / level)) * N) + 0.1)), max(1, int(np.round((outlier + (1 - outlier) * (i / level)) * N) + 0.1))
                vd.append(np.mean(vsort[:, pos[0]:pos[1]], axis=1))
                vu.append(np.mean(vsort[:, (-pos[1]):(-pos[0])], axis=1))
            else:
                pos = max(1, int(np.round((outlier + (1 - outlier) * (i / level)) * N) + 0.1))
                vd.append(np.mean(vsort[:, :pos], axis=1))
                vu.append(np.mean(vsort[:, (-pos):], axis=1))
        # Draw distributions
        mark_every = np.round(np.linspace(0, len(x) - 1, 10)).astype(np.int).tolist()
        ax.plot(x, avg, marker=m, ms=7, markevery=mark_every, **kwargs)
        for i in range(level):
            ax.fill_between(x, vd[i], vu[i], alpha=0.2, color=c['color'])
    __plot_configs(xlabel=xlabel, ylabel=ylabel, figure_size=figure_size,
                   has_legend=has_legend, legend_loc=legend_loc, legend_col=legend_col,
                   fig=fig, ax=ax)
