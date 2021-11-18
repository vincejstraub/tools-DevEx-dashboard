#### Column descriptions

* **Animal ID**: identification number assigned to each animal being tracked, currently equivalent to serial number of camera used to film trajectory.
* **Tank**: tank in which animal being tracked is located.
* **Experimental group**: experimental group that animal being tracked belongs to.
* **Tracking time (min)**: total tracking time in minutes for which data points exist combined across all segments for animal. N.B. this does not account for data points that are the result of tracking errors.
* **Data points (#)**: total number of rows with values for both x and y-coordinates. N.B. This may include erroneous  data points (i.e. outliers or 'spikes'). 
* **Data points removed (#)**: total number of rows that have been removed. N.B. This includes both data points lost during data collection (see _Dropped frames_ in Table 3) and those excluded during processing (i.e. outliers or 'spikes') due to random and or systematic error (see _Detected spikes_ and _Expected spikes_ in Table 3). 
* **Total activity (cm)**: cumulative step count, or the straight line distance from the beginning to the end point of the trajectory.
* **Mean activity (cm/unit time)**: mean activity per unit time, where unit time is set to hour by default.
* **Treatment (min)**: amount of feeding time, defined by linear mapping of activity values across animals, defined in the [`lbt_experiment`](https://github.com/vincejstraub/developing-exploration-behavior/blob/master/Processing/libratools/libratools/lbt_experiment.py) module.