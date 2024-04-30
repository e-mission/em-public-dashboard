import pandas as pd
import numpy as np
import arrow
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

sns.set_style("whitegrid")
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

# Module for pretty-printing outputs (e.g. head) to help users
# understand what is going on
# However, this means that this module can only be used in an ipython notebook

import IPython.display as disp


SAVE_DIR="/plots/"

def calculate_pct(labels, values):
    v2l_df = pd.DataFrame({"vals": values}, index=labels)

    # Calculate % for all the values
    vs = v2l_df.vals.sum()
    v2l_df["pct"] = v2l_df.vals.apply(lambda x: round((x/vs) * 100, 1))

    return (v2l_df.index.to_list(),v2l_df.vals.to_list(), v2l_df.pct.to_list())

def merge_small_entries(labels, values):
    v2l_df = pd.DataFrame({"vals": values}, index=labels)

    # Calculate % for all the values
    vs = v2l_df.vals.sum()
    v2l_df["pct"] = v2l_df.vals.apply(lambda x: round((x/vs) * 100, 1))
    disp.display(v2l_df)

    # Find small chunks to combine
    small_chunk = v2l_df.where(lambda x: x.pct <= 2).dropna()
    misc_count = small_chunk.sum()

    v2l_df = v2l_df.drop(small_chunk.index)
    disp.display(v2l_df)

    # This part if a bit tricky
    # We could have already had a non-zero other, and it could be small or large
    if "Other" not in v2l_df.index:
        # zero other will end up with misc_count
        v2l_df.loc["Other"] = misc_count
    elif "Other" in small_chunk.index:
        # non-zero small other will already be in misc_count
        v2l_df.loc["Other"] = misc_count
    else:
        # non-zero large other, will not already be in misc_count
        v2l_df.loc["Other"] = v2l_df.loc["Other"] + misc_count
    disp.display(v2l_df)

    return (v2l_df.index.to_list(),v2l_df.vals.to_list(), v2l_df.pct.to_list())

def process_data_frame(df, df_col):
    """ Inputs:
    df = Likely expanded_ct, data_eb or expanded_ct_sensed data frame
    df_col = Column from the above df, likely Mode_confirm, primary_mode
    trip_type = Bar labels (e.g. Labeled by user (Confirmed trips))
    """
    try:
        labels = df[df_col].value_counts(dropna=True).keys().tolist()
        values = df[df_col].value_counts(dropna=True).tolist()
        return process_trip_data(labels, values)
    except KeyError:
        print(f"Column '{df_col}' not found in the data frame.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(), pd.DataFrame()

def process_distance_data(df, df_col, distance_col, label_units_lower):
    """ Inputs:
    df = Likely expanded_ct, data_eb or expanded_ct_sensed data frame
    df_col = Column from the above df, likely Mode_confirm, primary_mode
    distance_col = Column associated with distance from above data frame
    label_units_lower = lbs/kg
    trip_type = Bar labels (e.g. Labeled by user (Confirmed trips))
    """
    try:
        dist = df.groupby(df_col).agg({distance_col: ['sum', 'count', 'mean']})
        dist.columns = ['Total (' + label_units_lower + ')', 'Count', 'Average (' + label_units_lower + ')']
        dist = dist.reset_index()
        dist = dist.sort_values(by=['Total (' + label_units_lower + ')'], ascending=False)

        dist_dict = dict(zip(dist[df_col], dist['Total (' + label_units_lower + ')']))
        labels_dist = []
        values_dist = []

        for x, y in dist_dict.items():
            labels_dist.append(x)
            values_dist.append(y)

        return process_trip_data(labels_dist, values_dist)
    except KeyError:
        print(f"Column '{df_col}' not found in the data frame.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(), pd.DataFrame()

def process_data_for_cutoff(df, df_col, distance_col):
    """ Inputs:
    df = Likely expanded_ct, data_eb or expanded_ct_sensed data frame
    df_col = E.g. Column from the above df, likely Mode_confirm, primary_mode
    distance_col = Column associated with distance from above data frame
    trip_type = Bar labels (e.g. Labeled by user (Confirmed trips))
    """
    try:
        cutoff = df.distance.quantile(0.8)
        if pd.isna(cutoff):
            cutoff = 0

        dist_threshold = df[distance_col].quantile(0.8).round(1)
        dist_threshold = str(dist_threshold)

        labels = df.loc[(df['distance'] <= cutoff)][df_col].value_counts(dropna=True).keys().tolist()
        values = df.loc[(df['distance'] <= cutoff)][df_col].value_counts(dropna=True).tolist()
        processed_data_expanded, processed_data = process_trip_data(labels, values)

        return processed_data_expanded, processed_data, cutoff, dist_threshold
    except KeyError:
        print(f"Column '{df_col}' not found in the data frame.")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame(), pd.DataFrame(), None, None

# Create dataframes with cols: 'Label' 'Value' and 'Proportion'
def process_trip_data(labels, values):
    """ Inputs:
    labels = Displayed labels (e.g. "Gas car, drove alone")
    values = Corresponding vlaues of these labels
    trip_type = Bar labels (e.g. Labeled by user (Confirmed trips))
    Returns:
    df_total_trip_expanded = Data frame without consolidation of Others, used to create the alt_html table
    df_total_trip = Data frame with consolidation of Others, used to represent the Bar Charts.
    """
    if len(labels) == 0 and len(values) == 0:
        return pd.DataFrame(), pd.DataFrame()
    m_labels_expanded, m_values_expanded, m_pct_expanded = calculate_pct(labels, values)
    data_trip_expanded = {'Label': m_labels_expanded, 'Value': m_values_expanded, 'Proportion': m_pct_expanded}
    df_total_trip_expanded = pd.DataFrame(data_trip_expanded)

    m_labels, m_values, m_pct = merge_small_entries(labels, values)
    data_trip = {'Label': m_labels, 'Value': m_values, 'Proportion': m_pct}
    df_total_trip = pd.DataFrame(data_trip)
    return df_total_trip_expanded, df_total_trip

# Creates/ Appends single bar to the 100% Stacked Bar Chart
def plot_stacked_bar_chart(df, bar_name, bar_lab, ax, colors_combined):
    """ Inputs:
    df = Data frame corresponding to the bar in a stacked bar chart
    bar_name = Text to represent in case data frame is empty (e.g. "Sensed Trip")
    bar_lab = Text to represent the Bar (e.g. Labeled by user\n (Confirmed trips))
    ax = axis information
    colors_combined = color mapping dictionary
    """
    sns.set(font_scale=1.5)
    bar_height = 0.2
    bar_width = [0]
    if df.empty:
        ax.text(x = 0.5, y = 0.5, s = f"No data available for {bar_name}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=20)
        ax.yaxis.set_visible(False)
    else:
        for label in pd.unique(df['Label']):
            long = df[df['Label'] == label]
            if not long.empty:
                mode_prop = long['Proportion']
                mode_count = long['Value']
                vals_str = [f'{y:.1f} %\n({x:.0f})' if y > 4 else '' for x, y in zip(mode_count, mode_prop)]
                bar = ax.barh(y=bar_lab, width=mode_prop, height=bar_height, left=bar_width, label=label, color=colors_combined[label])
                ax.bar_label(bar, label_type='center', labels=vals_str, rotation=90, fontsize=16)
                bar_width = [total + val for total, val in zip(bar_width, mode_prop)]
            else:
                print(f"{long} is empty")
        ax.tick_params(axis='y', labelsize=18)
        ax.tick_params(axis='x', labelsize=18, rotation=90)
        ax.legend(bbox_to_anchor=(1, 1), loc='upper left', fancybox=True, shadow=True, fontsize=15)
        # Fix for the error: RuntimeError("Unknown return type"), adding the below line to address as mentioned here https://github.com/matplotlib/matplotlib/issues/25625/
        ax.set_xlim(right=ax.get_xlim()[1] + 1.0, auto=True)

# Adds chart title, x and y axis label to the 100% Stacked Bar Chart
def add_stacked_bar_chart_title(fig, ax, plot_title, file_name):
    # Setup label and title for the figure since these would be common for all sub-plots
    fig.supxlabel('Proportion (Count)', fontsize=20, x=0.5, y= ax.xaxis.get_label().get_position()[0] - 0.62, va='top')
    fig.supylabel('Trip Types', fontsize=20, x=-0.12, y=0.5, rotation='vertical')
    fig.suptitle(plot_title, fontsize=25,va = 'bottom')
    plt.text(x=0, y=ax.xaxis.get_label().get_position()[0] - 0.62, s=f"Last updated {arrow.get()}", fontsize=12)
    plt.subplots_adjust(hspace=0.1, top= 0.95)
    fig.savefig(SAVE_DIR + file_name + ".png", bbox_inches='tight')
    plt.show()

def energy_impact(x,y,color,plot_title,file_name):
    color = color.map({True: 'green', False: 'red'})
    objects = ('Energy Savings', 'Energy Loss')
    
    y_labels = y
    plt.figure(figsize=(15, 8))
    width = 0.8
    ax = x.plot(kind='barh',width=width, color=color)
    ax.set_title(plot_title, fontsize=18)
    ax.set_xlabel('Energy_Impact(kWH)', fontsize=18)
    ax.set_ylabel('Replaced Mode',fontsize=18)
    ax.set_yticklabels(y_labels)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.relim()
    ax.autoscale_view() 

    rects = ax.patches
    for rect in rects:
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        space = 5
        ha = 'left'
       
        if x_value < 0:
            space *= -1
            ha = 'right'
        
        label = "{:.1f}".format(x_value)

        # Create annotation
        plt.annotate(
            label,                      
            (x_value, y_value),         
            xytext=(space, 0),          
            textcoords="offset points", 
            va='center',                
            ha=ha, fontsize=12, color='black', fontweight='bold')
        
    # map names to colors
    cmap = {True: 'green', False: 'red'}  
    patches = [Patch(color=v, label=k) for k, v in cmap.items()]
    plt.text(0,-1.5,f"Last updated {arrow.get()}", fontsize=10)
    plt.legend(labels=objects, handles=patches, loc='upper right', borderaxespad=0, fontsize=15, frameon=True)
    plt.savefig(SAVE_DIR+file_name+".png", bbox_inches='tight')

def barplot_mode(data,x,y,plot_title, labels, file_name):
    colours = dict(zip(labels, plt.cm.tab20.colors[:len(labels)]))
    sns.set(font_scale=1.5)
    f = plt.subplots(figsize=(15, 6))
    sns.set(style='whitegrid')
    ax = sns.barplot(x=x, y=y, palette=colours,data=data, ci=None)
    plt.xlabel(x, fontsize=23)
    plt.ylabel(y, fontsize=23)
    plt.title(plot_title, fontsize=25)
    # y should be based on the max range + the biggest label ("Gas Car, with others")
    plt.text(0,-(data[y].max()/8 + 3.3),f"Last updated {arrow.get()}", fontsize=10)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.savefig(SAVE_DIR+file_name+".png", bbox_inches='tight')

def barplot_mode2(data,x,y,y2,plot_title,file_name):
    all_labels= ['Gas Car, drove alone',
                 'Bus', 
                 'Train', 
                 'Free Shuttle',
                 'Taxi/Uber/Lyft', 
                 'Gas Car, with others', 
                 'Bikeshare',
                 'Scooter share',
                 'E-bike', 
                 'Walk', 
                 'Skate board', 
                 'Regular Bike', 
                 'Not a Trip',
                 'No Travel', 
                 'Same Mode', 
                 'E-car, drove alone',
                 'E-car, with others',
                 'Air',
                 'Other']
    
    colours = dict(zip(all_labels, plt.cm.tab20.colors[:len(all_labels)]))
    sns.set(font_scale=1.5)

    fig, ax1 = plt.subplots(figsize=(15,6))
    #bar plot creation
    ax1.set_title(plot_title, fontsize=16)
    plt.text(0,-2,f"Last updated {arrow.get()}", fontsize=10)
    ax1.set_xlabel(x, fontsize=16)
    ax1.set_ylabel(y, fontsize=16)
    ax1 = sns.barplot(x=x, y=y, data = data, palette=colours, ci=None)
    ax1.grid(False)
    
    #specify we want to share the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:red'
    #line plot creation
    ax2.set_ylabel('Count', fontsize=16)
    ax2 = sns.pointplot(x=x, y=y2, data = data, sort=False, color=color)
    ax2.grid(False)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.savefig(SAVE_DIR+file_name+".png", bbox_inches='tight')
    
def barplot_day(data,x,y,plot_title,file_name):
    sns.set(font_scale=1.5)
    f = plt.subplots(figsize=(15, 6))
    sns.set(style='whitegrid')
    ax = sns.barplot(x=x, y=y,data=data, ci=None, color='blue')
    plt.xlabel(x, fontsize=16)
    plt.ylabel(y, fontsize=16)
    plt.title(plot_title, fontsize=16)
    # heuristic where we take the max value and divide it by 8 to get the scale
    # the 8 is heuristic based on experimentation with the CanBikeCO data
    plt.text(0,-(data[y].max())/8,f"Last updated {arrow.get()}", fontsize=10)
    plt.savefig(SAVE_DIR+file_name+".png", bbox_inches='tight')

def CO2_impact(x,y,color,plot_title, xLabel, yLabel, file_name):
    color = color.map({True: 'green', False: 'red'})
    objects = ('CO2 Reduction', 'CO2 Increase')

    y_labels = y
    plt.figure(figsize=(15, 8))
    width = 0.8
    ax = x.plot(kind='barh',width=width, color=color)
    ax.set_title(plot_title, fontsize=18)
    ax.set_xlabel(xLabel, fontsize=18)
    ax.set_ylabel(yLabel,fontsize=18)
    ax.set_yticklabels(y_labels)
    ax.xaxis.set_tick_params(labelsize=15)
    ax.yaxis.set_tick_params(labelsize=15)
    ax.relim()
    ax.autoscale_view()

    rects = ax.patches
    for rect in rects:
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2
        space = 5
        ha = 'left'

        if x_value < 0:
            space *= -1
            ha = 'right'

        label = "{:.1f}".format(x_value)

        # Create annotation
        plt.annotate(
            label,
            (x_value, y_value),
            xytext=(space, 0),
            textcoords="offset points",
            va='center',
            ha=ha, fontsize=12, color='black', fontweight='bold')

    # map names to colors
    cmap = {True: 'green', False: 'red'}
    patches = [Patch(color=v, label=k) for k, v in cmap.items()]
    plt.text(0,-1.5,f"Last updated {arrow.get()}", fontsize=10)
    plt.legend(labels=objects, handles=patches, loc='upper right', borderaxespad=0, fontsize=15, frameon=True)
    plt.savefig(SAVE_DIR+file_name+".png", bbox_inches='tight')

def timeseries_plot(x,y,plot_title,ylab,file_name):
    fig, ax = plt.subplots(figsize=(16,4))
    sns.lineplot(ax=ax, x=x, y=y).set(title=plot_title, xlabel='Date', ylabel=ylab)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.25)
    ax.figure.savefig(SAVE_DIR+file_name+".png", bbox_inches='tight')

def timeseries_multi_plot(data,x,y,hue,plot_title,ylab,legend_title,file_name):
    fig, ax = plt.subplots(figsize=(16,4))
    sns.lineplot(ax=ax, data=data, x=x, y=y, hue=hue).set(title=plot_title, xlabel='Date', ylabel=ylab)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.25)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='best', borderaxespad=0, title=legend_title)
    ax.figure.savefig(SAVE_DIR+file_name+".png", bbox_inches='tight')

def access_alt_text(alt_text, chart_name, write_permission='w'):
    """ Inputs:
    alt_text = the text describing the chart
    chart_name = the alt text file to save or update
    """
    f = open(SAVE_DIR+chart_name+".txt", write_permission)
    f.write(alt_text)
    f.close()
    return alt_text

def store_alt_text_generic(chart_description, chart_name, var_name):
    """ Inputs:
    chart_description = what type of chart is it
    chart_name = what to label chart by in the dictionary
    var_name = the variable being analyzed across the chart
    """
    # Fill out the alt text based on components of the chart and passed data
    alt_text = f"{chart_description} chart of {var_name}."
    alt_text = access_alt_text(alt_text, chart_name)
    return alt_text

def store_alt_text_bar(df, chart_name, var_name):
    """ Inputs:
    df = dataframe with index of item names, first column is counts
    chart_name = what to label chart by in the dictionary
    var_name = the variable being analyzed across pie slices
    """
    # Fill out the alt text based on components of the chart and passed data
    alt_text = f"Bar chart of {var_name}."
    for i in range(0,len(df)):
        alt_text += f" {df.index[i]} is {np.round(df.iloc[i,0], 1)}."
    alt_text = access_alt_text(alt_text, chart_name)
    return alt_text

# Appends bar information into the alt_text file
def store_alt_text_stacked_bar_chart(df, chart_name, var_name):
    """ Inputs:
    df = dataframe combining columns as Trip Type, Label, Value, Proportion
    chart_name = name of the chart
    """
    # Generate alt text file
    alt_text = f"\nStacked Bar of: {var_name}\n"
    for i in range(len(df)):
        alt_text += f"{df['Label'].iloc[i]} is {df['Value'].iloc[i]}({df['Proportion'].iloc[i]}%).\n"
    alt_text = access_alt_text(alt_text, chart_name, 'a')

    return alt_text

def store_alt_text_timeseries(df, chart_name, var_name):
    """ Inputs:
    df = dataframe with first col of dates, second column is values
    chart_name = what to label chart by in the dictionary
    var_name = the variable being analyzed across pie slices
    """
    # Fill out the alt text based on components of the chart and passed data
    alt_text = f"Scatter chart of {var_name}."
    arg_min = np.argmin(df.iloc[:,1])
    arg_max = np.argmax(df.iloc[:,1])
    alt_text += f" First minimum is {np.round(df.iloc[arg_min,1], 1)} on {df.iloc[arg_min,0]}. First maximum is {np.round(df.iloc[arg_max,1], 1)} on {df.iloc[arg_max,0]}."
    alt_text = access_alt_text(alt_text, chart_name)
    return alt_text

# Creating html table with col as Trip Type, Label, Value, and Proportion
def access_alt_html(html_content, chart_name, write_permission):
    """ Inputs:
    html_body = the text describing the chart
    chart_name = the alt text file to save or update
    var_name = the variable being analyzed across bars
    """
    with open(SAVE_DIR + chart_name + ".html", f'{write_permission}') as f:
        f.write(html_content)

    return html_content

# Appends bar information into into the alt_html
def store_alt_html_stacked_bar_chart(df, chart_name,var_name):
    """ Inputs:
    df = dataframe combining columns as Trip Type, Label, Value, Proportion
    chart_name = name of the chart
    """
    # Generate html table
    alt_html = "\n"
    for i in range(len(df)):
        alt_html += f"<tr><td>{df['Label'].iloc[i]}</td><td>{df['Value'].iloc[i]}</td><td>{df['Proportion'].iloc[i]}%</td></tr>"
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <body>
        <p>Trip Type: {var_name}</p>
        <table border="1" style="background-color: white;">
            <tr>
                <th>Label</th>
                <th>Value</th>
                <th>Proportion</th>
            </tr>
            {alt_html}
        </table>
    </body>
    </html>
    """
    alt_html = access_alt_html(html_content, chart_name, 'a')

    return alt_html

# Creates the html file, and appends plot_title
def create_alt_html_title(plot_title, chart_name, missing_text=""):
    """ Inputs:
    plot_title = Overall plot title
    chart_name = name of the chart
    missing_text = Text to indicate missing data
    """
    plot_title += f"\n {missing_text}"
    alt_html = f"""
    <!DOCTYPE html>
    <html>
    <body>
        <p>{plot_title}</p>
    </body>
    </html>
    """
    alt_html = access_alt_html(alt_html, chart_name, 'w')

    return alt_html

# Creates the alt text file, and appends the plot_title
def create_alt_text_title(plot_title, chart_name, missing_text=""):
    """ Inputs:
    plot_title = Overall plot title
    chart_name = name of the chart
    missing_text = Text to indicate missing data
    """
    # if not missing_text:
    plot_title += f"\n {missing_text}"
    alt_text = access_alt_text(plot_title, chart_name, 'w')

    return alt_text

def generate_missing_plot(plot_title,debug_df,file_name):
    f, ax = plt.subplots(figsize=(10,10))

    plt.title("Unable to generate plot\n"+plot_title+"\n Reason:", fontsize=25, color="red")
    # Must keep the patch visible; otherwise the entire figure becomes transparent
    # f.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    # ax = sns.barplot(x=debug_df['count'],y=debug_df.index, palette=sns.color_palette("Reds",n_colors=10))
    # ax.set_xlim(0, None)
    # for i in ax.containers:
    #     ax.bar_label(i,)
    the_table = plt.table(cellText=debug_df.values,
              rowLabels=debug_df.index,
              colLabels=debug_df.columns,
              loc="center")
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(20)
    the_table.scale(1, 4)
    cellDict = the_table.get_celld()
    for i in range(1,len(debug_df)+1):
        currCellTextStr = cellDict[(i,0)].get_text().get_text()
        currCellTextFloat = float(currCellTextStr)
        if np.isnan(currCellTextFloat):
            cellDict[(i,0)].get_text().set_text("None")
        if np.isnan(currCellTextFloat) or currCellTextFloat == 0:
            cellDict[(i, 0)].get_text().set_color("red")
    plt.savefig(SAVE_DIR+file_name+".png", bbox_inches='tight')

def store_alt_text_missing(df, chart_name, var_name):
    """ Inputs:
    df = dataframe with index of debug information, first column is counts
    chart_name = what to label chart by in the dictionary
    var_name = the variable being analyzed across pie slices
    """
    # Fill out the alt text based on components of the chart and passed data
    alt_text = f"Unable to generate\nBar chart of {var_name}.\nReason:"
    for i in range(0,len(df)):
        alt_text += f" {df.index[i]} is {np.round(df.iloc[i,0], 1)}."
    alt_text = access_alt_text(alt_text, chart_name)
    return alt_text
