import numpy as np
from matplotlib import colors
import mpl_scatter_density
import matplotlib.pyplot as plt
# import plotly.express as px
import warnings
import seaborn as sns
import fcsparser

#This is largely copied from /home/jj2765/DAmFRET_denoising/tomato/DAmFRETClusteringTools/ClusterPlotting.py, but with modifications desired by Hannah.


#add arial
from matplotlib import font_manager
arialPath = "/n/projects/jj2765/mEos_nanobody/fonts/arial.ttf"
font_manager.fontManager.addfont(arialPath)

# Set the default font size and family
#does this change the defaults for plots that don't use this module, but do import it?
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 10

def plotDAmFRETCluster(x, y, color="blue", ax=None, title=None, vmin=0.25, xlab="mEos3 concentration (p.d.u.)", ylab="AmFRET"):
    """
    Inspired by Tayla's function and the "double" example from the mpl_scatter_density README
    If ax is not given, the only plot will be the cluster plot
    """
    if not ax:
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(1,1,1, projection= 'scatter_density')

    #     ax.scatter_density(x, y, color=color) #only shows most dense part
    #     ax.scatter_density(x, y, color=color, norm=colors.LogNorm()) #blocky
    #setting dpi in scatter_density the same as figure dpi leads to swirls and other fun problems.
    if len(x) >= 2: #only one cell breaks things because vmin = vmax
        ax.scatter_density(x, y, color=color, norm=colors.LogNorm(), vmin=vmin, vmax=np.nanmax, dpi=100, clip_on=True) #I think this looks good, clip_on added on 6/13 to better address saturated acceptor values
    #update: clip_on doesn't actually help.
    #increasing zorder allows the plot to lay over the spines, but that shouldn't be necessary, the spines should be outside of the data.
    #Because they aren't actually outside of the data, I set the right spine to be 0.5 further out. That seems to be the width of a point in scatter_density, but I'm not sure if it's flexible across dpis and figsizes.
    #ax.scatter_density(x, y, color=color, norm=colors.LogNorm(), vmin=vmin, vmax=np.nanmax, dpi=100, zorder=5)
    #ax.scatter_density(x, y, color=color, norm=colors.LogNorm(), vmin=vmin, vmax=np.nanmax, dpi=None, zorder=5)

    ax.spines.right.set_position(("outward", 0.5))
    ax.set_xlabel(xlab, loc="center")
    ax.set_ylabel(ylab, loc="center")
    if title:
        ax.set_title(title)
    return ax

def plotDAmFRETDensity(x, y, ax=None, vmin=0.25, title=None, xlab="mEos3 concentration (p.d.u.)", ylab="AmFRET", returnFig=False, xlims=None, ylims=None, logX=False, mpl_dpi=100):
    """
    Generates a density plot for all cells in a well.
    Based on Tayla's function.
    If ax is not given, the only plot will be the density plot
    xlims and ylims must be a tuple of (min, max), or will be automatically determined if None
    """
    if not ax:
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(1,1,1, projection= 'scatter_density')
    if xlims is None:
      xlims = (min(x), max(x))
    if ylims is None:
        ylims = (min(y), max(y))
        if type(y).__name__ == "Series":
            if y.name == "AmFRET":
                ylims = (-0.2,2) #for this paper, use this AmFRET range by default
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    norm = colors.LogNorm()
    cmap = plt.get_cmap("viridis")
    cmap.set_under(alpha=0)

    ax.scatter_density(x, y, norm=norm, vmin=vmin, vmax=np.nanmax, cmap=cmap, dpi=mpl_dpi, clip_on=True) #BAD
    #ax.scatter_density(x, y, norm=norm, vmin=vmin, vmax=np.nanmax, cmap=cmap, dpi=100, zorder=5)
    #ax.scatter_density(x, y, norm=norm, vmin=vmin, vmax=np.nanmax, cmap=cmap, dpi=None, zorder=5)

    ax.spines.right.set_position(("outward", 0.5))
    ax.set_xlabel(xlab, loc="center")
    ax.set_ylabel(ylab, loc="center")
    if title:
        ax.set_title(title)

    yTicks = np.arange(-10,15.1, 0.5) #way larger range than will likely ever be seen
    #values are -10, -9.5, -9, ... but only ticks within the AmFRET bounds will be shown
    #selecting within range isn't necessary because we set ylim after this
    ax.set_yticks(yTicks)

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)



    if logX:
        ax.set_xscale("log")

    #if y contains a pd.Series and y axis is AmFRET, add the dashed line at 0
    #checks type because y could be a list or other non pd.Series type.
    #This could fail if a Series in a different module is provided, but I think that's unlikely
    #Even polars series seem to use the <Series>.name convention
    if type(y).__name__ == "Series":
        if y.name == "AmFRET":
            ax.hlines(0, xlims[0], xlims[1], color=(0.6,0.6,0.6), linestyle="--")
    
    if returnFig:
      return fig
    return ax

def plotDAmFRETClusters(x, y, labels, ax=None, colors=plt.get_cmap("tab10").colors, vmin=0.25, figWidth=4.5, figHeight=3, title=None, xlab="mEos3 concentration (p.d.u.)", ylab="AmFRET", returnFig=False, xlims=None, ylims=None, logX=False, labelOrder=None):
    """
    Generates a single figure with subplots of the whole well density and cluster densities.
    colors takes an iterable of colors (names or iterable of rgb)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #I've only ever seen warnings in mpl_scatter_density due to empty bins, which we expect

        if not ax:
            fig = plt.figure(figsize=(figWidth,figHeight), dpi=150)
            ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

        uniqueLabels, labelCounts = np.unique(labels, return_counts=True)
        labelCountDict = dict(zip(uniqueLabels, labelCounts))

        #start positions - will be relative to size of axis, not data
        textX = 0.01
        textYMax = 0.98

        if labelOrder is None:
            #make the label for the highest pop
            label = uniqueLabels.max()
            plotDAmFRETCluster(x[labels == label], y[labels == label], colors[label], ax, vmin=vmin, xlab=xlab, ylab=ylab)
            higherText = ax.text(textX, textYMax, f"{labelCountDict[label]}, {labelCountDict[label] / sum(labelCountDict.values()):.2f}", va="top", ha="left", color=colors[label], transform = ax.transAxes)
            
            #make label for non-highest pop(s)
            for label in uniqueLabels[-2::-1]:
                plotDAmFRETCluster(x[labels == label], y[labels == label], colors[label], ax, vmin=vmin, xlab=xlab, ylab=ylab)
                
                higherText = ax.annotate(f"{labelCountDict[label]}, {labelCountDict[label] / sum(labelCountDict.values()):.2f}", xycoords=higherText, xy=(0,-1), color=colors[label], horizontalalignment="left", transform = ax.transAxes)

        else:
            label = labelOrder[0]
            
            plotDAmFRETCluster(x[labels == label], y[labels == label], colors[label], ax, vmin=vmin, xlab=xlab, ylab=ylab)
            higherText = ax.text(textX, textYMax, f"{labelCountDict[label]}, {labelCountDict[label] / sum(labelCountDict.values()):.2f}", va="top", ha="left", color=colors[label], transform = ax.transAxes)
            
            #make label for non-highest pop(s)
            for label in labelOrder[1:]:
                plotDAmFRETCluster(x[labels == label], y[labels == label], colors[label], ax, vmin=vmin, xlab=xlab, ylab=ylab)
                
                higherText = ax.annotate(f"{labelCountDict[label]}, {labelCountDict[label] / sum(labelCountDict.values()):.2f}", xycoords=higherText, xy=(0,-1), color=colors[label], horizontalalignment="left", transform = ax.transAxes)
        
        if xlims is None:
            xlims = (min(x), max(x))
        if ylims is None:
            ylims = (min(y), max(y))
            if type(y).__name__ == "Series":
                if y.name == "AmFRET":
                    ylims = (-0.2,2) #for this paper, use this AmFRET range by default
                    
        yTicks = np.arange(-10,15.1, 0.5) #way larger range than will likely ever be seen
        #values are -10, -9.5, -9, ... but only ticks within the AmFRET bounds will be shown
        #selecting within AmFRET range isn't necessary because we set ylim after this
        ax.set_yticks(yTicks)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        if logX:
            ax.set_xscale("log")

        #if y contains a pd.Series and y axis is AmFRET, add the dashed line at 0
        #checks type because y could be a list or other non pd.Series type.
        #This could fail if a Series in a different module is provided, but I think that's unlikely
        #Even polars series seem to use the <Series>.name convention
        if type(y).__name__ == "Series":
            if y.name == "AmFRET":
                ax.hlines(0, xlims[0], xlims[1], color=(0.6,0.6,0.6), linestyle="--")

        if title:
            ax.set_title(title)

        #fig.tight_layout()
        if returnFig:
            return fig
        return ax

def plotDAmFRETDensityAndClusters(x, y, labels, colors=plt.get_cmap("tab10").colors, vmin=0.25, figWidth=9, figHeight=3, title=None, xlab="mEos3 concentration (p.d.u.)", ylab="AmFRET", xlims=None, ylims=None, logX=False):
    """
    Generates a single figure with subplots of the whole well density and cluster densities.
    colors takes an iterable of colors (names or iterable of rgb)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #I've only ever seen warnings in mpl_scatter_density due to empty bins, which we expect

        fig = plt.figure(figsize=(figWidth,figHeight), dpi=150)
        ax = fig.add_subplot(1, 2, 1, projection='scatter_density')
        plotDAmFRETDensity(x, y, ax, vmin=vmin, xlab=xlab, ylab=ylab, xlims=xlims, ylims=ylims, logX=logX)

        if xlims is None:
            xlims = (min(x), max(x))
        if ylims is None:
            ylims = (min(y), max(y))
            if type(y).__name__ == "Series":
                if y.name == "AmFRET":
                    ylims = (-0.2,2) #for this paper, use this AmFRET range by default
                    
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        ax = fig.add_subplot(1, 2, 2, projection='scatter_density')
        plotDAmFRETClusters(x,y,labels, ax, colors, vmin=vmin, xlab=xlab, ylab=ylab, xlims=xlims, ylims=ylims, logX=logX)

        #if y contains a pd.Series and y axis is AmFRET, add the dashed line at 0
        #checks type because y could be a list or other non pd.Series type.
        #This could fail if a Series in a different module is provided, but I think that's unlikely
        #Even polars series seem to use the <Series>.name convention
        if type(y).__name__ == "Series":
            if y.name == "AmFRET":
                ax.hlines(0, xlims[0], xlims[1], color=(0.6,0.6,0.6), linestyle="--")

        if title:
            fig.suptitle(title)

        fig.tight_layout()
        return fig

def plotBDFPAcceptorContours(dataToPlot, labelColumn="manualLabel", colors=plt.get_cmap("tab10").colors, ax=None, xlims=None, ylims=None, returnFig=False):
    if not ax:    
        fig = plt.figure()
        ax = plt.gca()

    if "BDFP/SSC" not in dataToPlot:
        dataToPlot["BDFP/SSC"] = dataToPlot["BDFP1.6-A"] / dataToPlot["SSC 488/10-A"]

    colorsToUse = {label: colors[label] for label in dataToPlot[labelColumn].unique()}

    #palette can take a dictionary, so this should work
    sns.kdeplot(dataToPlot, x="Acceptor/SSC", y="BDFP/SSC", hue=labelColumn, palette=colorsToUse, fill=True, alpha=0.5, legend=False, log_scale=True, ax=ax, clip=(xlims, ylims))

    defaultXLabel = "mEos3 concentration (p.d.u.)"
    defaultYLabel = "BDFP1.6:1.6 concentration (p.d.u.)"

    ax.set_xlabel(defaultXLabel)
    ax.set_ylabel(defaultYLabel)
    
    if returnFig:
        return fig

def plotBDFPAcceptorContours_test(dataToPlot, labelColumn="manualLabel", colors=plt.get_cmap("tab10").colors, ax=None, xlims=None, ylims=None, returnFig=False, hueOrder=None):
    if not ax:    
        fig = plt.figure()
        ax = plt.gca()

    if "BDFP/SSC" not in dataToPlot:
        dataToPlot["BDFP/SSC"] = dataToPlot["BDFP1.6-A"] / dataToPlot["SSC 488/10-A"]

    colorsToUse = {str(label): colors[label] for label in dataToPlot[labelColumn].unique()}
    if hueOrder is not None:
        stringHueOrder = [str(item) for item in hueOrder]
    else:
        stringHueOrder = None

    #palette can take a dictionary, so this should work
    # sns.kdeplot(dataToPlot, x="Acceptor/SSC", y="BDFP/SSC", hue=labelColumn, palette=colorsToUse, fill=True, alpha=0.5, legend=False, log_scale=True, ax=ax, clip=(xlims, ylims))
    sns.kdeplot(dataToPlot, x="Acceptor/SSC", y="BDFP/SSC", hue=dataToPlot[labelColumn].astype(str), palette=colorsToUse, fill=True, alpha=0.5, legend=False, log_scale=True, ax=ax, clip=(xlims, ylims), hue_order=stringHueOrder)

    defaultXLabel = "mEos3 concentration (p.d.u.)"
    defaultYLabel = "BDFP1.6:1.6 concentration (p.d.u.)"

    ax.set_xlabel(defaultXLabel)
    ax.set_ylabel(defaultYLabel)
    
    if returnFig:
        return fig

def readDataToDF(filename, minAmFRET=-0.2, maxAmFRET=1.0, minAmFRETPercentile=0.01, maxAmFRETPercentile = 99.99, minAcceptorPercentile=0.1, maxAcceptorPercentile=100, minLDAPercentile=0, maxLDAPercentile=100, xAxis ="log(Acceptor)"):
    """
    Reads an FCS file and returns a dataframe. Also computes AmFRET, logAcceptor, and logDonor/Acceptor.
    Drops na values, filters extreme acceptor and amfret values.
    Filtration occurs in the order: 1. AmFRET value, 2. Acceptor percentile, 3. AmFRET percentile, 4. LDA percentile
    If xAxis is "log(Acceptor/SSC)", that column will be made in addition to "log(Acceptor)". Only those two values are allowed for xAxis.
    Note: this function should probably be replaced at some point to allow for more axis options and more flexibility in general.
    Also, this function should be made to filter combinatorically, not serially (there are probably better terms, but hopefully you know what I mean).
    Defaults may be removed at some point to emphasize making the config file specific.
    """

    if xAxis not in ["log(Acceptor)", "log(Acceptor/SSC)"]:
        raise NotImplementedError("xAxis can only be either 'log(Acceptor)', or 'log(Acceptor/SSC)'")

    #metadata, data = fcsparser.parse(filename)
    metadata, data = fcsparser.parse(filename)


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        #np.log10 produces a lot of warnings, but NAs get removed, so we can ignore them

        data["AmFRET"] = data["FRET-A"] / data["Acceptor-A"]
        data["Donor/Acceptor"] = data["Donor-A"] / data["Acceptor-A"]

        data["logDonor/Acceptor"] = np.log10(data["Donor/Acceptor"])
        data["log(Donor/Acceptor)"] = np.log10(data["Donor/Acceptor"])

        data["logAcceptor"] = np.log10(data["Acceptor-A"])
        data["log(Acceptor)"] = np.log10(data["Acceptor-A"])

        if xAxis == "log(Acceptor/SSC)":
            if "SSC 488/10-A" not in data.columns:
                raise ValueError("SSC channel not found in data!")
            else:
                data["Acceptor/SSC"] = data["Acceptor-A"] / data["SSC 488/10-A"]
                data["log(Acceptor/SSC)"] = np.log10(data["Acceptor/SSC"])

        if len(data) < 7:
        # I don't like that this is hard coded, but this is only meant to avoid exceptions that break loops
        # I think this should always work?
            return data.dropna()

        #This is probably very inneficient because it makes tons of copies of data.
        #generating multiple arrays of logicals, combining them together, then applying to data would be much better.
        
        data = data.dropna()

        data = data[data["AmFRET"] >= minAmFRET]
        data = data[data["AmFRET"] <= maxAmFRET]

        if xAxis == "log(Acceptor)":
            acceptor_bounds = np.nanpercentile(data["log(Acceptor)"], (minAcceptorPercentile,maxAcceptorPercentile))
            data = data[data["log(Acceptor)"] >= acceptor_bounds[0]]
            data = data[data["log(Acceptor)"] <= acceptor_bounds[1]]
        elif xAxis == "log(Acceptor/SSC)":
            acceptor_bounds = np.nanpercentile(data["log(Acceptor/SSC)"], (minAcceptorPercentile,maxAcceptorPercentile))
            data = data[data["log(Acceptor/SSC)"] >= acceptor_bounds[0]]
            data = data[data["log(Acceptor/SSC)"] <= acceptor_bounds[1]]


        AmFRET_bounds = np.nanpercentile(data["AmFRET"], (minAmFRETPercentile,maxAmFRETPercentile))
        data = data[data["AmFRET"] >= AmFRET_bounds[0]]
        data = data[data["AmFRET"] <= AmFRET_bounds[1]]

        LDA_bounds = np.nanpercentile(data["logDonor/Acceptor"], (minLDAPercentile,maxLDAPercentile))
        data = data[data["logDonor/Acceptor"] >= LDA_bounds[0]]
        data = data[data["logDonor/Acceptor"] <= LDA_bounds[1]]

        return data.dropna()