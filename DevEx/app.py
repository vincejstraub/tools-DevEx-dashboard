import os    # standard imports
import time
import pathlib
import datetime

import yaml    # 3rd party packages
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from libratools import lbt_utils    # local imports
from libratools import lbt_datasets
from libratools import lbt_experiment


CONFIG_PATH = pathlib.Path.cwd().parent / './config.yml'
WARNING_MSG = ' appears to have a large number of tracking errors.'

def main():
    """
    Load pages.
    """
    logo_path  = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / DASHBOARD_DOCS_DIR / 'logo.png'
    logo = Image.open(logo_path)
    st.sidebar.image(logo, output_format='PNG')
    st.sidebar.title("Control panel")
    st.sidebar.header('Navigation')
    side_menu_selectbox = st.sidebar.selectbox('', ['Home', 'About', 'Contact'])
    if side_menu_selectbox == 'Home':
        homepage_path = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / DASHBOARD_DOCS_DIR / 'homepage.md'
        configure_page_text(text_path=homepage_path)
        how_to_load, data_path, date, cameras_to_load, file_type_arg = configure_homepage_sidebar('Data access')
        skip_track_arg, save_table_arg, filter_arg = configure_homepage_sidebar_further(
            'Further settings')
        if data_path is None:
            with st.beta_expander("Getting started instructions"):
                instructions_path = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / DASHBOARD_DOCS_DIR / 'instructions.md'
                configure_page_text(text_path=instructions_path)
                st.info('Tip: Try selecting the sample data option.')
            pass
        else:
            if how_to_load == 'Sample data':
                with st.spinner('Loading sample data...'):
                        time.sleep(2)
                        configure_homepage_display(how_to_load, data_path, None, None, None, None)
                        time.sleep(1)
                        st.balloons()
            else:
                agree = st.checkbox(
                    f'Proceed with loading data from disk recorded on {date} using current further settings?')
                if agree:
                    st.subheader('Notifications')
                    with st.spinner(f'Loading data from disk...'):
                        if file_type_arg == 'merged':
                            configure_basic_homepage_display(how_to_load, data_path, cameras_to_load,
                                                             file_type_arg)
                        elif file_type_arg == 'processed':
                            configure_homepage_display(how_to_load, data_path, skip_track_arg,
                                                       save_table_arg, cameras_to_load, 
                                                       file_type_arg)                           
    elif side_menu_selectbox == 'About':
        about_path = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / DASHBOARD_DOCS_DIR / 'aboutpage.md'
        configure_page_text(
            text_path=about_path)
    elif side_menu_selectbox == 'Contact':
        contact_path = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / DASHBOARD_DOCS_DIR / 'contactpage.md'
        configure_page_text(text_path=contact_path)

    
def configure_basic_homepage_display(how_to_load, data_path, cameras, file_type):
    """
    Configure basic homepage main display.
    """
    # load data
    data, warnings = load_data(how_to_load, data_path, None, 
                               cameras, file_type)
    # treatment and activity summary
    st.header('Simple Trajectory Overview')
    configure_table(
        data, '', 'Table 1: Trajectory overview',
        cols=OVERVIEW_TABLE1_DIC.keys(), metadata_dic=OVERVIEW_TABLE1_DIC,
        new_cols=OVERVIEW_TABLE1_NEW_COLS)
    # display trajectory monitor section
    st.header('Individual monitor')
    for traj in data.keys():
        st.markdown(f"### Animal ID: {data[traj]['metadata']['camera_id']}")
        visualize_trajectory(data[traj]['data'])
        
    
def configure_homepage_display(how_to_load, data_path, skip_track_arg, 
                               save_table_arg, cameras, file_type):
    """
    Configure homepage main display.
    """
    # load data
    data, warnings = load_data(
        how_to_load, data_path, skip_track_arg, cameras, file_type)
    # display notifications
    if skip_track_arg == False and len(warnings) >= 1:
        for cam, msg in warnings.items():
            st.error(msg)
    if save_table_arg == None:
        pass
    elif len(save_table_arg) != 0:
        for table in save_table_arg:
            st.success(f'{table.capitalize()} saved to file.')
    # activity summary
    st.header('Trajectory metrics')
    configure_table(
        data, '', 'Table 1: Activity and tracking duration',
        cols=TREATMENT_TABLE1_DIC.keys(), metadata_dic=TREATMENT_TABLE1_DIC,
        new_cols=TREATMENT_TABLE1_NEW_COLS, update_cols=True, 
        save_table=save_table_arg)
    with st.beta_expander('Column description', expanded=False):
        table1_description_path = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / DASHBOARD_DOCS_DIR / 'table1_description.md'
        configure_page_text(table1_description_path)
    configure_table(
        data, '', 'Table 2: Primary movement metrics',
        cols=DIAGNOSTIC_TABLE1_DIC.keys(), metadata_dic=DIAGNOSTIC_TABLE1_DIC,
        new_cols=DIAGNOSTIC_TABLE1_NEW_COLS, 
        save_table=save_table_arg)
    with st.beta_expander('Column description', expanded=False):
        table2_description_path = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / DASHBOARD_DOCS_DIR / 'table2_description.md'
        configure_page_text(table2_description_path)
    # display diagnostic metrics
    st.header('Error Diagnosis')
    configure_table(
        data, '', 'Table 3: Additional diagnostic metrics across trajectories',
        cols=DIAGNOSTIC_TABLE2_DIC.keys(), metadata_dic=DIAGNOSTIC_TABLE2_DIC,
        new_cols=DIAGNOSTIC_TABLE2_NEW_COLS, 
        save_table=save_table_arg)
    with st.beta_expander('Column description', expanded=False):
        table3_description_path = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / DASHBOARD_DOCS_DIR / 'table3_description.md'
        configure_page_text(table3_description_path)
    # display trajectory monitor section
    st.header('Individual monitor')
    for traj in data.keys():
        st.markdown(f"### Animal ID: {data[traj]['metadata']['camera_id']}")
        visualize_trajectory(data[traj]['data'])
        visualize_primary_metrics(df=data[traj]['data'], bar_col='stepLength', 
                                  activity_interval=20, angle_interval=10)
        visualize_direction(data[traj]['data'])
        with st.beta_expander('Figure description', expanded=False):
            docs_path = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / DASHBOARD_DOCS_DIR
            figure_description =  docs_path / 'figure_description.md'
            configure_page_text(figure_description)


def configure_homepage_sidebar(title):
    """
    Configure primary sidebar options.
    """
    # declare empty variables
    cameras_to_load = None
    file_type = None
    # set page settings
    st.sidebar.header(title)
    how_to_load = st.sidebar.selectbox(
        'How to access data: ',
        ('<select>', 'Sample data', 'From disk', 'Upload'))
    if how_to_load == '<select>':
        data_option, date = None, None
    elif how_to_load == 'Upload':
        data_option = st.sidebar.file_uploader("Choose a CSV file", type='.csv')
        date = None
    elif how_to_load == 'URL':
        data_option = st.sidebar.text_input('File URL: ')
        if data_option == '':
            st.sidebar.warning('No data found, try inputting a different URL.')
    elif how_to_load == 'Sample data':
        data_option = DASHBOARD_DATA_DIR
        date = lbt_utils.strptime_date_arg('20210314') + ' collected in pilot study'
    elif how_to_load == 'From disk':
        # get yesterday's date by default
        yesterdate = lbt_utils.get_date(delta=-1)
        yyyy, mm, dd = yesterdate[:4], yesterdate[4:6], yesterdate[-2:]
        date_input = st.sidebar.date_input(
            "Date for which to load data: ",
            datetime.date(int(yyyy), int(mm), int(dd)))
        date = date_input.strftime('%Y-%m-%d')
        data_option = date
        # add option to select only certain directories
        cameras_path = HOME_DIR / REPO_ROOT_DIR / 'Processing/' / CAMERAS_IDS
        cameras = lbt_datasets.read_yaml_as_dict(cameras_path)
        camera_ids = [cameras[i]['ID'] for i in cameras.keys()]
        cameras_to_load = st.sidebar.multiselect(
            "Cameras for which to load data: ",
            ['all cameras'] + camera_ids, ['all cameras'])
        file_type = st.sidebar.selectbox(
            'File type to load:',
            ('merged', 'processed'), 1)

    return how_to_load, data_option, date, cameras_to_load, file_type


def configure_homepage_sidebar_further(subheading):
    """
    Configure sidebar for homepage.
    """
    st.sidebar.header(subheading)
    save_table_arg = st.sidebar.multiselect(
        'Tables to save to file:', 
        ['Table 1', 'Table 2', 'Table 3'])
    skip_track_arg = st.sidebar.selectbox(
        'Skip trajectories that appear to be corrupted:', 
        ('only show warning', 'skip track if threshold surpassed'))
    if skip_track_arg == 'only show warning':
        skip_track = False
    elif skip_track_arg == 'skip track if threshold surpassed':
        skip_track = True
    filter_arg = st.sidebar.selectbox(
      'Apply a Kalman filter to noisy tracks:', 
      ('no', 'yes'))
    if filter_arg == 'yes':
        filter_tracks = True
    elif filter_arg == 'no':
        filter_tracks = False
    
    return skip_track, save_table_arg, filter_tracks


def configure_table(table, text, figure_title, cols='',
                    metadata_dic='', new_cols='', update_cols=False, 
                    save_table=None):
    """
    Stylize and display data and text for section one.
    """
    # display text
    st.write(text)
    # add a subheader for table
    st.markdown(figure_title)
    if update_cols is True:
        # update and tabulate table dict
        table = tabulate_metadata_table(
            table, metadata_dic, new_cols)
    else:
        table = describe_metadata(
            table, cols=cols,
            metadata_dic=metadata_dic, new_cols=new_cols)
    # style and display table
    st.write(styled_table(table))

    # save table to file 
    if save_table == save_table:
        pass
    elif figure_title.split(':')[0] in save_table:    
        table_name = figure_title.split(':')[0].replace(' ', '_').lower()
        date_today = lbt_utils.get_date(delta=0)
        file_path = DASHBOARD_OUTPUTS_DIR + date_today + f'_{table_name}.csv'
        open(file_path, 'w').write(table.to_csv())


def tabulate_metadata_table(data, cols, new_cols):
    """
    Compute treatment times, describe metadata, and tabulate results.
    """
    # get activity values
    cams_activity = lbt_utils.get_nested_dict_values(data)
    
    # get suggested treatments
    treatments, start_times = lbt_experiment.get_feeding_times(
        cams_activity, min_time=2, max_time=180,
        end_time='12:00:00', round_min=False)

    # zip and add to metadata
    add_to_metadata = lbt_utils.zip_lists(
        data.keys(), treatments, start_times)
    data = lbt_utils.add_nested_vals_to_dict(
        data, add_to_metadata)
    
    # return table
    return describe_metadata(data, cols=cols.keys(),
                             metadata_dic=cols,
                             new_cols=new_cols)


def describe_metadata(data, metadata_col='metadata',
                      cols='', metadata_dic='',
                      new_cols=''):
    """
    Returns pandas.DataFrame of metadata.   
    """
    # declare DataFrame to display
    df = pd.DataFrame([metadata_dic])
    
    # append key metrics columns for each trajectory dict to df
    for file in data.keys():
        metrics_dic = data[file][metadata_col]

        for col in metrics_dic:
            row = {col: metrics_dic[col] for col in cols}
        df = df.append(row, ignore_index=True)

    # drop first row of filler values
    df = df.iloc[1:]

    # set column dtypes
    df = df.astype(metadata_dic)

    # rename cols
    df.rename(columns=dict(zip(cols, new_cols)), inplace=True)
    
    return df


def load_data(how_to_load, data_path, skip_track_arg, camera_ids, file_type):
    """
    Check whether to load data via path argument or directly.
    """
    if how_to_load == '<select>':
        pass
    elif how_to_load == 'Upload':
        pass
    elif how_to_load == 'Sample data':
        data = load_sample_data(data_path)
        warnings = None
    elif how_to_load == 'From disk':
        # load camera ids
        if camera_ids[0] == 'all cameras':
            cameras_path = HOME_DIR / REPO_ROOT_DIR / 'Processing/' / CAMERAS_IDS
            cameras = lbt_datasets.read_yaml_as_dict(cameras_path)
            camera_ids = [cameras[i]['ID'] for i in cameras.keys()]
        data, warnings = load_data_from_disk(
            data_path, camera_ids, skip_track_arg, file_type)

    return data, warnings


@st.cache(allow_output_mutation=True)
def load_sample_data(datadir):
    """
    Load sample data.
    """
    # locate sample BioTracker files
    indir = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / datadir
    file_paths, _ = lbt_datasets.read_file_paths(
        indir=indir,
        warning=False)
    
    # iterate over file paths
    data_dict = {}
    for file in file_paths:
        # load BioTracker segment into DataFrame
        df, _ = lbt_datasets.load_trajectory(
            file, skiprows=33, sep=',', cols=PROCESSED_COLS,
            keycols=['globalFRAME', 'x', 'y'])
        comments = lbt_datasets.read_metadata(file, num_comment_lines=33)
        metadata = lbt_datasets.extract_comments_as_dict(comments)
        for k in metadata.keys():
            if k == 'source_fps':
                try:
                    metadata[k] = float(metadata[k].split(':')[-1].strip())
                except AttributeError:
                    pass
        key = pathlib.Path(file).stem
        data_dict[key] = {'data': df, 'metadata': metadata}

    return data_dict


def load_data_from_disk(date, camera_ids, skip_track_arg, file_type):
    """
    Load data.
    """
    # strip date argument of - char
    date = date.replace('-', '')
    # validate date argument
    if len(date) > 8:
        date = ''.join(date.split('-'))
    else:
        date = date
    # declare which columns to use
    if file_type == 'processed':
        cols = PROCESSED_COLS
        comment_lines = 33
    else:
        cols = MERGED_COLS
        comment_lines = 7
    # load data for each camera
    all_data_dict = {}
    # collect corrupt track warnings
    corrupt_tracks = {}
    for cam in list(camera_ids):
        # declare directory path based on date argument
        date_dir = lbt_datasets.find_dir(
            path=DATA_DIR,
            prefix=date, suffix=cam)
        
        if date_dir is None:
            raise_error(f'Error: No data found for camera {str(cam)} on {date}.')
            st.info('Tip: Try locating any missing data or removing the camera \
                     from camera_ids.yaml, then rerun the dashboard.')
        
        # declare trajectory segments path based on date argument and camera
        segments_dir = DATA_DIR / str(cam) / date_dir
        
        # locate BioTracker files
        file_paths, _ = lbt_datasets.read_file_paths(
            indir=segments_dir,
            suffix=True,
            suffix_str='_' + file_type,
            warning=False)
        
        # iterate over file paths
        for file in file_paths:
            # load BioTracker segment into DataFrame pending skip corrupt tracks arg
            comments = lbt_datasets.read_metadata(file, num_comment_lines=comment_lines)
            if skip_track_arg == True:
                if str(comments[-1].split(':')[-1].strip()) != 'positive':
                    df, metadata = load_csv(file, comments, cols, comment_rows=comment_lines)
                    key = pathlib.Path(file).stem 
                    all_data_dict[key] = {'data': df, 'metadata': metadata}
            else:
                if str(comments[-1].split(':')[-1].strip()) == 'positive': 
                    df, metadata = load_csv(file, comments, cols, comment_rows=comment_lines)
                    key = pathlib.Path(file).stem 
                    all_data_dict[key] = {'data': df, 'metadata': metadata}
                    corrupt_tracks[key] = 'Camera ' + str(cameras[cam]['ID']) + WARNING_MSG                    
                else: 
                    df, metadata = load_csv(file, comments, cols, comment_rows=comment_lines)
                    key = pathlib.Path(file).stem 
                    all_data_dict[key] = {'data': df, 'metadata': metadata}
    
    return all_data_dict, corrupt_tracks


def load_csv(file, comments, cols, key_cols=['globalFRAME', 'x', 'y'], 
             comment_rows=29):
    """
    Load CSV from disk.
    """
    df, _ = lbt_datasets.load_trajectory(
        file, skiprows=comment_rows, sep=',', cols=cols,
        keycols=key_cols)
    metadata = lbt_datasets.extract_comments_as_dict(comments)

    return df, metadata


def raise_error(error_message):
    st.error(error_message)


def configure_page_text(text_path):
    """
    Format page and load text.
    """
    with open(text_path, 'r', encoding='utf-8') as text:
        page_text = text.read()
        st.markdown(page_text, unsafe_allow_html=True)


def styled_table(df):
    """
    Style pandas.DataFrame.
    """
    df = df.style.set_properties(**{'background-color': 'black',
                                    'color': 'lawngreen',
                                    'border-color': 'black'})
    return df


def visualize_direction(trajectory, angle_col='turnAngle'):
    """
    Returns distribution of change in direction as a polar chart.
    """
    # round turning angles 
    df = trajectory[[angle_col]].round(0)
    # exclude rows with no change in turning angle
    df = df.dropna()
    # count unique occurences 
    df = df.groupby([angle_col]).size().reset_index(name='Count')
    # plot 
    fig = px.bar_polar(df, r='Count', theta=angle_col,
                       color='Count', 
                       template='plotly_dark',
                       range_theta=[-180, 180], start_angle=90, direction='clockwise', 
                       range_r=[0,0.1],
                       color_discrete_sequence= px.colors.sequential.Viridis,
                       title='sfss')

    fig.update_layout(
        polar=dict(radialaxis=dict(showticklabels=False, ticks='', linewidth=0)),
        title={
        'text': 'E) Relative orientation',
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        width=500,
        height=500
    )
    
    # edit annotations
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=16)

    # return plot
    st.plotly_chart(fig, use_container_width=True)
    

def visualize_trajectory(trajectory, xvals='x', yvals='y', animate='chunk_segment', 
                         line_color=['#68FC00'], xlabel='x-coordinate (cm)', 
                         ylabel='y-coordiante (cm)', title=''):
    """
    Return scatter plot of each trajectory segment.
    """
    fig = px.line(trajectory, x=xvals, y=yvals, animation_frame=animate, 
                  color_discrete_sequence=line_color)

    # update xaxis properties
    fig.update_xaxes(title_text=xlabel, nticks=10, showline=True, 
                     linewidth=1, linecolor='white', mirror=True, 
                     range=[-10, 90], constrain='domain')
    fig.update_yaxes(title_text=ylabel, nticks=10, showline=True,
                     linewidth=1, linecolor='white', mirror=True,
                     scaleanchor='x', scaleratio=1)

    # update title and height
    fig.update_layout(template='plotly_dark',
                      showlegend=False, 
                      width=850,
                      height=700,
                      title={
                        'text': f'A) Trajectory of animal movement',
                        'y':0.96,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},)
    
    # edit annotations
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=16)
    
    # return plot
    st.plotly_chart(fig, use_container_width=True)
    
    
def visualize_primary_metrics(df, bar_col='stepLength', activity_interval=60,
                              angle_interval=10, angle_col='turnAngle'):
    """
    Returns distribution of step lengths, activity and relative turning
    angles as 2 x 2 subplot.
    """
    # compute presets
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    num_boxes = len([np.asarray(i[1].round(0)) for i in list(
        df.groupby([pd.Grouper(key='timestamp', freq='10min')])[angle_col])])

    # initialize figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.35, 0.65],
        row_heights=[0.5, 0.5],
        shared_xaxes='rows',
        specs=[[{'rowspan': 2}, {'rowspan': 1}],
               [{}, {'rowspan': 1}]], 
        subplot_titles=(
            'B) Distribution of step lengths', 
            'C) Change in mean activity',
            '',
            'D) Change in distribution of relative turning angles'))

    # add or append traces
    fig.add_trace(
        go.Histogram(x=df.stepLength, marker_color='RoyalBlue'), row=1, col=1)
    
    fig.add_trace(
        go.Bar(
            x=list(df.groupby([pd.Grouper(
                key='timestamp', 
                freq=f'{activity_interval}min')]).sum().index.strftime('%H:%M')),
            y=list(df.groupby([pd.Grouper(
                key='timestamp', freq=f'{activity_interval}min')]).sum()[bar_col]),
            marker={'color': list(
                df.groupby([pd.Grouper(
                    key='timestamp', freq=f'{activity_interval}min')]).sum()[bar_col]),
                    'colorscale': 'Viridis'}), row=1, col=2)

    boxplots = [go.Box(
        y=[np.asarray(i[1].round(0)) for i in list(
            df.groupby([pd.Grouper(
                key='timestamp', freq=f'{angle_interval}min')])[angle_col])][i],
        name=[i[0].strftime('%H:%M') for i in list(
            df.groupby([pd.Grouper(
                key='timestamp', freq=f'{angle_interval}min')])[angle_col])][i],
        boxpoints=False,
        marker=dict(
            color='rgb(147,196,125)',
            line=dict(
                outliercolor='rgba(219, 64, 82, 0.6)',
                outlierwidth=2)),
        line_color='rgb(230,145,56)'
        ) for i in range(int(num_boxes))]
    for trace in boxplots:
        fig.append_trace(trace, row=2, col=2)


    # update xaxis properties
    fig.update_xaxes(title_text='Step length (cm)', nticks=10, 
                    showline=True, linewidth=1, linecolor='white', mirror=True, 
                    row=1, col=1)
    fig.update_xaxes(title_text='Time interval ', nticks=10, 
                    showline=True, linewidth=1, linecolor='white', mirror=True, 
                    row=1, col=2)
    fig.update_xaxes(title_text='Time interval', nticks=10, 
                    showline=True, linewidth=1, linecolor='white', mirror=True, 
                    row=2, col=2)

    # update yaxis properties
    fig.update_yaxes(title_text='Count (log)', type="log",
                    showline=True, linewidth=1, linecolor='white', mirror=True, 
                    row=1, col=1)
    fig.update_yaxes(title_text='Activity (cm)', nticks=10, 
                    showline=True, linewidth=1, linecolor='white', mirror=True, 
                    row=1, col=2)
    fig.update_yaxes(title_text='Angle (°)', nticks=10, 
                    showline=True, linewidth=1, linecolor='white', mirror=True, 
                    row=2, col=2)

    # add shapes
    fig.add_shape(
        type="line", x0=5, y0=0.72, x1=5, y1=7000, 
        line=dict(color="red", width=1, dash="dash"), 
        row=1, col=1)

    # edit annotations
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=16)

    # update title and height
    fig.update_layout(showlegend=False, 
        template="plotly_dark",
        margin=dict(r=10, t=25, b=40, l=60),
        width=800,
        height=500
    )

    # show fig
    st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    # read config file
    with open(CONFIG_PATH) as c:
        config = yaml.safe_load(c)
    # store directory paths for DevEx global vars
    DATA_DIR = pathlib.Path(config['PATHS']['DATA_DIR'])
    REPO_ROOT_DIR = config['PATHS']['REPO_ROOT_DIR']
    DASHBOARD_DIR = config['PATHS']['DASHBOARD_DIR']
    DASHBOARD_DATA_DIR = config['PATHS']['DASHBOARD_DATA_DIR']
    DASHBOARD_DOCS_DIR = config['PATHS']['DASHBOARD_DOCS_DIR']
    DASHBOARD_IMAGES_DIR = config['PATHS']['ISSUE_LINK']
    ISSUE_LINK = config['PATHS']['ISSUE_LINK']
    OVERVIEW_TABLE1_NEW_COLS = config['VARS']['OVERVIEW_TABLE1_NEW_COLS']
    OVERVIEW_TABLE1_DIC = config['VARS']['OVERVIEW_TABLE1_DIC']
    DIAGNOSTIC_TABLE1_DIC = config['VARS']['DIAGNOSTIC_TABLE1_DIC']
    DIAGNOSTIC_TABLE1_NEW_COLS = config['VARS']['DIAGNOSTIC_TABLE1_NEW_COLS']
    DIAGNOSTIC_TABLE2_DIC = config['VARS']['DIAGNOSTIC_TABLE2_DIC']
    DIAGNOSTIC_TABLE2_NEW_COLS = config['VARS']['DIAGNOSTIC_TABLE2_NEW_COLS']
    # read directory paths for project file vars
    MERGED_COLS = config['VARS']['LIBRATOOLS_MERGED_COLS']
    PROCESSED_COLS = config['VARS']['LIBRATOOLS_PROCESSED_COLS']
    CAMERAS_IDS = config['FILES']['CAMERAS_IDS']    
    TREATMENT_TABLE1_NEW_COLS = config['VARS']['TREATMENT_TABLE1_NEW_COLS']
    TREATMENT_TABLE1_DIC = config['VARS']['TREATMENT_TABLE1_DIC']
    DASHBOARD_OUTPUTS_DIR = config['PATHS']['DASHBOARD_OUTPUTS_DIR']
    # store home diretory where repo is saved
    HOME_DIR = pathlib.Path(os.getcwd()).parents[3]
    # configure main app display
    im_path = HOME_DIR / REPO_ROOT_DIR / DASHBOARD_DIR / DASHBOARD_DOCS_DIR / 'icon.png'
    icon = Image.open(im_path)
    st.set_page_config(
        page_title='DevEx Dashboard', layout='wide',
        page_icon=icon,
        initial_sidebar_state='auto')
#     try:
    main()
#     except:
#         st.error(f"Oops! Something went wrong... Please check your input.\
#                  If you think there is a bug, please open up an [issue] \
#                  ({ISSUE_LINK}).")
