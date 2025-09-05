# Discourse frames usage

We study the usage of frames across union and for specific unions in ``frames_usage.ipynb``

# Detrended usage computation
We aim at detrending frames usage scores before and after the elections. As such, we apply the following steps:
1. For each union, we generate periods of 18 days in which there is no event of a given kind (losing or winning elections) with ``generate_empty_periods.py``.
2. In ``compute_baseline_prop.py`` we compute proportion aggregates for the usage of frames (to avoid sparsity) and we further aggregate such proportions in before and after for baseline periods.
3. We compute detrended scores in ``compute_deltas.py``, computing differences between actual and baseline scores before and after an event.

# Pre-event and change post-event analysis
We analyze the usage of frames before an election in ``pre_event_comparison.ipynb`` to compare pre-election usage.
We then focus on the change that happens after the elections. As such, we first cluster changes into increase, decrease and stable patterns in ``slope_analysis.py`` and then we analyze offsets between losing and winning with ``pattern_analysis_slopes.ipynb``.
