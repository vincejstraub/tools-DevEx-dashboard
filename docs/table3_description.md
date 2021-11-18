#### Column descriptions

* **Animal ID**: identification number assigned to each animal being tracked, currently set to serial number of camera used to film trajectory.
* **Tank**: tank in which animal being tracked is located.
* **Source FPS**: frequency (rate) at which consecutive images called frames are recorded, set in config.ini stored in the directory [/Processing](https://github.com/vincejstraub/developing-exploration-behavior/tree/master/Processing).
* **Dropped frames (#)**: frames dropped by loopbio automatically during recording.
* **Detected spikes (#)**: number of data points classified as bivariate outliers or 'spikes', i.e, possesing suspicious values for ‘x’ and ‘y’ coordinate values. These are expected to occur both for random and systematic reasons.
* **Expected spikes (#)**: number of data points in the first user-defined time interval (default=1 second) across all segments classified as spikes and excluded due exclusively to expected systematic error. These are expected to occur, for instance, when the tracking software tracks a dirt particle for a single frame and then returns to the animal.
* **Incomplete rows (#)**: number of rows with one or more column values labelled as ‘NaN’ (not a number) or containing empty strings.
* **Missing x-coords (#)**: number of rows with x-coordinate values labelled as ‘NaN’ (not a number) or containing empty strings.
* **Missing y-coords (#)**: number of rows with y-coordinate values labelled as ‘NaN’ (not a number) or containing empty strings.
