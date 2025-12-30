# Discourse frames usage

We study the usage of frames across union and for specific unions in ``frames_usage.ipynb``

## Focus on emotions and topics

### Emotions
1. Apply pre-trained emotion classifier to get the distribution of emotion probabilities within Facebook posts with ``emotion_detection.py``.
2. Analyze correlation between frames and emotions with ``emotion_analysis.ipynb``.
3. Compute moving aggregation for emotion with ``compute_rolling_emotions.py``.

### Topics
1. Topic modeling on all Facebook posts with ``topic_modeling.py``.
2. Topic modeling on two example unions with ``topic_modeling_byunion.ipynb``.

# Detrended usage computation
We aim at detrending frames usage scores before and after the elections. As such, we apply the following steps:
1. For each union, we generate periods of 18 days in which there is no event of a given kind (losing or winning elections) with ``generate_empty_periods.py``.
2. In ``compute_baseline_prop.py`` we compute proportion aggregates for the usage of frames (to avoid sparsity) and we further aggregate such proportions in before and after for baseline periods.
3. We compute detrended scores in ``compute_deltas.py``, computing differences between actual and baseline scores before and after an event.

# Predicting election outcomes
1. Save data to be used as input for modeling with ``savedata_for_bayesian_modeling.py``.
2. Fit Bayesian logistic regression model with ``bayesian_model.py``.
3. Plot coefficients obtained from model fit with ``bayesian_coefficients_plot.ipynb``.

# Pre- vs. post-event analysis
We then focus on the change that happens after the elections. As such, we first cluster changes into increase, decrease and stable patterns in ``slope_analysis.py`` and then we analyze offsets between losing and winning with ``pattern_analysis_slopes.ipynb``.
