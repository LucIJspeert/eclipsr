# ECLIPSR
### Eclipse Candidates in Light curves and Inference of Period at a Speedy Rate


## What is ECLIPSR?
ECLIPSR is a Python code that is aimed at the automatic analysis of space based light curves to find eclipsing binaries and provide some first order measurements, among others the binary star period and eclipse depths. It provides a recipe to find individual eclipses using the time derivatives of the light curves, with as primary benefit that eclipses can still be detected in light curves of stars where the dominating variability is for example pulsations. Since the algorithm detects each eclipse individually, even light curves containing only one eclipse can in principle be successfully analysed (and classified). 
The aim is to find eclipsing binaries among both pulsating and non-pulsating stars in a homogeneous and quick manner to be able to process large amounts of light curves in reasonable amounts of time. The output includes, but is not limited to, the individual eclipse markers, the period and time of first (primary) eclipse and a score between 0 and 1 indicating the likelihood that the analysed light curve is that of an eclipsing binary. See the documentation for more information on the discriminating power of this variable.


### Reference Material

* This algorithm has been documented, tested and applied in the publication: [paper](https://webpage.com/something/123456789/8044)


## Getting started
As of version 1.0 (January 2021), the way to get this code is to either download it or make a fork on GitHub. It is recommended to download the latest release from the GitHub page. To use it, simply import the folder containing ECLIPSR into your own script.

**ECLIPSR has only been tested in Python 3.7**. 
Using older versions could result in unexpected errors, although any Python version >3.6 is expected to work.

**Package dependencies:** The following package versions have been used in the development of this code, meaning older versions can in principle work, but this is not guaranteed. NumPy 1.17.3, SciPy 1.5.2, Numba 0.51.2, Matplotlib 3.1.1, Astropy (for .fits functionality), h5py (for saving results). Newer versions are expected to work, and it is considered a bug if this is not the case.

### Example use

Since the main feature of ECLIPSR is its fully automated operation, taking advantage of its functionality is as simple as running one or two functions:

	>>> import eclipsr as ecl
	>>> # it is recommended to run this before using other functions (see its function description)
	>>> times, signal = ecl.utility.ingest_signal(times, signal, tess_sectors=True)
	>>> # find_eclipses() combines all of the functionality into one function
	>>> t_0, period, conf, sine_like, n_kernel = ecl.find_eclipses(times, signal, mode=1, tess_sectors=True)

The tess_sectors argument is a TESS space satellite-specific function, to be used if multiple sectors of TESS data are ingested at once. 
The signal ingest function converts the usual TESS data (electron counts) into a median normalised time series where nan-values are removed. If your input time series is not in counts, please make sure it is median normalised (non-negative) and only contains finite values before using the find\_eclipses() function.

To get more output than just the five listed in the above example, use different mode numbers. Mode 2 will give every last bit of possibly useful information that was determined from the analysis.

	>>> result = ecl.find_eclipses(times, signal, mode=2, tess_sectors=True)
	>>> t_0, period, score, sine_like, wide, n_kernel, width_stats, depth_stats, ecl_mid, widths, depths, ratios, added_snr, ecl_indices, flags_lrf, flags_pst = result

### Explanation of output

Here is a list of all possible outputs (as shown in the example above) and their meaning.

* t_0: the time of first (primary) eclipse. The first full eclipse is preferred if present (full meaning both the ingress and the egress where detected).

* period: the eclipse period detected. This is equal to the orbital period if either there are undetected secondary eclipses between the detected primary eclipses, or the secondary eclipses are detected and identified as such (by being different enough from the primaries).

* score: the score between 0 and 1, indicating if the analysed light curve might be of an eclipsing binary. Can also be -1 in case no eclipse detections were made at all. See the reference for more details (generally, a value of >~0.36 can be considered as a cut off point).

* sine_like: a boolean value indicating whether a signal might look like a sine curve. 

* wide: a boolean value indicating whether the average eclipse width is larger than 0.6 times the period.

* n_kernel: the integer width of the smoothing kernel used to convolve the signal. Higher values result in smoother curves, but also run the risk of deviating too far from the original signal. If tess_sectors=True, each sector will have its own value (and the output is an array).

* width_stats: a 2x2 array with the mean and standard deviation of the primary and secondary eclipse widths. width_stats[0] gives the primary eclipses, width_stats[:, 0] gives the means of both primary and secondary eclipses.

* depth_stats: same as width_stats but for the eclipse depths.

* ecl_mid: the times of midpoint for all detected eclipse candidates as they were measured from the derivatives.

* widths: the widths of all detected eclipse candidates.

* depths: the depths of all detected eclipse candidates.

* added_snr: a measure of the strength of the eclipse signal in both the original time series and its derivatives, used throughout the algorithm. Note: the name indicates that this can be interpreted as a signal-to-noise measure, and although it is roughly comparable and constructed in a similar way, this is not strictly the case! See the reference material for more about this quantity.

* flags_lrf: integer values indicating whether the eclipse candidate consists of a full eclipse, or a left (ingress) or right half (egress). See also the function: interpret_flags().

* flags_pst: integer values indicating whether the eclipse candidate is a primary or a secondary eclipse, or does not belong to the pattern that forms the detected period. This third category can contain any noise that was picked up, but also tertiary eclipses. See also the function: interpret_flags().

* ecl_indices: an array with 4 indices for each eclipse candidate, which mark the following points in the time series in order: the eclipse start, bottom left, bottom right and eclipse end. The two bottom points can coincide.


## Bugs and Issues

Despite all the testing, I am certain that there are still bugs in this code, or will be created in future versions. 

If you happen to come across any bugs or issues, *please* contact me. Only known bugs can be resolved.
This can be done through opening an issue on the ECLIPSR GitHub page: [LucIJspeert/eclipsr/issues](https://github.com/LucIJspeert/eclipsr/issues), or by contacting me directly (see below).

If you are (going to be) working on new or improved features, I would love to hear from you and see if it can be implemented in the source code.


## Contact

For questions and suggestions, please contact:

* luc.ijspeert(at)kuleuven.be

**Developer:** Luc IJspeert (KU Leuven)
