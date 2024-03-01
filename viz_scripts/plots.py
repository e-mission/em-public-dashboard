import pandas as pd
import numpy as np
import arrow
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from itertools import product

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

# Create dataframe with cols: 'Mode' 'Count' and 'Proportion'
def process_trip_data(labels, values, trip_type):
    m_labels, m_values, m_pct = calculate_pct(labels, values)
    data_trip = {'Mode': m_labels, 'Count': m_values, 'Proportion': m_pct}
    df_total_trip = pd.DataFrame(data_trip)
    df_total_trip['Trip Type'] = trip_type
    return df_total_trip

# Input: List of all dataframes
# Ouput: A single dataframe such that Trip Type has all Mode
def merge_dataframes(all_data_frames):
    # Concatenate DataFrames
    df = pd.concat(all_data_frames, ignore_index=True)

    # Create DataFrame with unique combinations of 'Trip Type' and 'Mode'
    unique_combinations = pd.DataFrame(list(product(df['Trip Type'].unique(), df['Mode'].unique())), columns=['Trip Type', 'Mode'])

    # Merge the original DataFrame with the unique combinations DataFrame
    merged_df = pd.merge(unique_combinations, df, on=['Trip Type', 'Mode'], how='left').fillna(0)
    return merged_df

def stacked_bar_chart_generic(plot_title, df, file_name, num_bars):
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(1,1, figsize=(15,6))
    width = 0.8

    running_total_long = [0] * num_bars

    mode_mapping = {
        "IN_VEHICLE": "IN_VEHICLE (Sensed)",
        "UNKNOWN": "UNKNOWN (Sensed)",
        "OTHER": "OTHER (Sensed)",
        "BICYCLING": "BICYCLING (Sensed)",
        "WALKING": "WALKING (Sensed)",
        "AIR_OR_HSR": "AIR_OR_HSR (Sensed)"
    }

    colors = plt.cm.tab20.colors[:len(pd.unique(df['Mode']))]

    for idx, mode in enumerate(pd.unique(df.Mode)):
        long = df[df['Mode'] == mode]

        if not long.empty:
            labels = long['Trip Type']
            vals = long['Proportion']
            bar_labels = long['Count']

            mode = mode_mapping.get(mode, mode)
            vals_str = [f'{y:.1f} %\n({x:.0f})' if y>4 else '' for x, y in zip(bar_labels, vals)]
            bar = ax.barh(labels, vals, width, left=running_total_long, label=mode, color = colors[idx])
            ax.bar_label(bar, label_type='center', labels=vals_str, rotation=90, fontsize=16)
            running_total_long = [total + val for total, val in zip(running_total_long, vals)]
        else:
            print(f"{mode} is unavailable.")
    ax.set_title(plot_title, fontsize=25)
    ax.set_xlabel('Proportion (Count)', fontsize=20)
    ax.set_ylabel('Trip Types', fontsize=20)
    ax.tick_params(axis='y', labelsize=18)
    ax.tick_params(axis='x', labelsize=18, rotation=90)
    # The Last updated text is placed just right below the X-axis
    plt.text(0,ax.xaxis.get_label().get_position()[0] - 1,f"Last updated {arrow.get()}", fontsize=12)

    ax.legend(bbox_to_anchor=(1, 1), fancybox=True, shadow=True)
    plt.subplots_adjust(bottom=0.25)
    fig.savefig(SAVE_DIR+file_name+".png", bbox_inches='tight')
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

def access_alt_text(alt_text, chart_name):
    """ Inputs:
    alt_text = the text describing the chart
    chart_name = the alt text file to save or update
    """
    f = open(SAVE_DIR+chart_name+".txt",'w')
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

def store_alt_text_stacked_bar_chart(df, chart_name, var_name):
    """ Inputs:
    df = dataframe combining columns as Trip Type, Mode, Count, Proportion
    chart_name = name of the chart
    var_name = the variable being analyzed across bars
    """
    # Generate alt text file
    alt_text = f"Stacked Bar chart of {var_name}."
    for i in range(len(df)):
        alt_text += f"Trip Type: {df['Trip Type'].iloc[i]} - Mode: {df['Mode'].iloc[i]} - Count: {df['Count'].iloc[i]} - Proportion: {df['Proportion'].iloc[i]}%\n"
    alt_text = access_alt_text(alt_text, chart_name)

    # Generate html table
    alt_html = ""
    for i in range(len(df)):
        alt_html += f"<tr><td>{df['Trip Type'].iloc[i]}</td><td>{df['Mode'].iloc[i]}</td><td>{df['Count'].iloc[i]}</td><td>{df['Proportion'].iloc[i]}%</td></tr>"
    alt_html = access_alt_html(alt_html, chart_name, var_name)

    return alt_text, alt_html

def store_alt_text_pie(df, chart_name, var_name):
    """ Inputs:
    df = dataframe with index of item names, first column is counts
    chart_name = what to label chart by in the dictionary
    var_name = the variable being analyzed across pie slices
    """
    # Fill out the alt text based on components of the chart and passed data
    alt_text = f"Pie chart of {var_name}."
    for i in range(0,len(df)):
        alt_text += f" {df.index[i]} is {np.round(df.iloc[i,0] / np.sum(df.iloc[:,0]) * 100, 1)}%."
    alt_text = access_alt_text(alt_text, chart_name)
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

# Creating html table with col as Trip Type, Mode, Count, and Proportion
def access_alt_html(alt_text, chart_name, var_name):
    """ Inputs:
    alt_text = the text describing the chart
    chart_name = the alt text file to save or update
    var_name = the variable being analyzed across bars
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{var_name}</title>
    </head>
    <body>
        <p>{var_name}</p>
        <table border="1" style="background-color: white;">
            <tr>
                <th>Trip Type</th>
                <th>Mode</th>
                <th>Count</th>
                <th>Proportion</th>
            </tr>
            {alt_text}
        </table>
    </body>
    </html>
    """
    with open(SAVE_DIR + chart_name + ".html", 'w') as f:
        f.write(html_content)

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
