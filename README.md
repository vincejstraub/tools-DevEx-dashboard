# DevEx 

[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active) [![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)

DevEx is a web-based dashboard to monitor daily tracking activity of animal movement trajectories. It can be used with any x, y, times CSV files but is specifically built for data collected using the [loopbio motif video recording system](http://loopbio.com/recording/) and labelled with [BioTracker](https://github.com/BioroboticsLab/biotracker_core), a computer vision framework for visual animal tracking. It relies on [libratools](https://github.com/vincejstraub/tools-libratools) for processing trajectories. 

It serves two main functions: 

1. It provides a quick read-out of movement metrics and tracking errors detected in recordings, thereby enabling a reset of camera settings, if necessary. 
2. It allows lab technicians to allocate treatments to animals (e.g. food to each fish tank based on each fish's daily activity).

For testing the dashboard, a number of sample tracks can currently be  read from the directory `data/`. As soon as you run the dashboard, you will be greeted with further instructions on how to access sample data or load data from disk.


## Installation and Setup

Install locally via `pip install -e . --user`. 

## Usage

`# streamlit run app.py`

## User Agreement

By downloading DevExDashboard you agree with the following points: DevExDashboard is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application of DevExDashboard.

## Maintenance

* [Vincent J. Straub](https://github.com/vincejstraub)  

## Requirements

Requirements are listed in the file :

* `requirements.txt`

Please follow  online instructions to install the required libraries, depending on your operating system and machine specifications.

