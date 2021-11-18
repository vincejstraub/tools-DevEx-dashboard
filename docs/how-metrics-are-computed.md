# How mean activity is computed

The below summarizes how mean activity (per time interval) is computed, which requires knowledge of how step length and total activity is defined. 

## Step length

The step length $d$ for a frame is displacement, the length of a line segment, between consecutive data points, or x-y coordinates. It is defined as the Euclidean distance between two points in Euclidean space:

$$d\left( p,q\right)   = \sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^ 2)}$$

where point $p$ has Cartesian coordinates ($p_1, p_2$) and point $q$ has coordinates ($q_1, q_2$).

## Total activity (cumulative step count)

Total activity is the cumulative step count, or the straight line distance from the beginning to the end point of the trajectory. Defined as:

$$\sum_{i=m}^{n}a_i = a_m + a_m+1 + ... + a_n-1 + a_n$$

where _i_ is the frame number; $a_i$ is the step length for frame $i$; $m$ is the lower bound of the frame count, and $n$ is the upper bound of the frame count.

## Mean activity

Mean activity is the average cumulative step count per time interval $t$ where time interval is by default set to 60 (minutes) and is computed as follows: first, the trajectory is split into $n$ sequences of equal length by dividing the total tracking duration $d$ by $t$; n.b. if $t$ is not a multiple of $d$, then some sequences may not be of equal length, i.e., shorter than $t$.

For each sequence of data points, total activity is then computed using the same formula above. Once a set of total activity values per time interval have been computed, mean activity (per time interval) is simply computed as the unweight arithmetic mean:

$$\bar{X} = \frac{\sum_{i=1}^{n} x_{i}}{n}$$


where $i$ is the sequence; $x_i$ is the total activity for sequence $i$; $m$ is the lower bound of the sequence count, and $n$ is the upper bound of the sequence cout. 