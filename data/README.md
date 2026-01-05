This folder contains some useful data sources to run the analysis. Intermediate data outputs are not shared but authors can be contacted for those, if needed.

The folder contains the following files:
* ``elections_nlrb_all2024.csv`` - file containing all data for elections held between Jan 2015 and Dec 2024 and reported on the NLRB website (note that also decertification elections are included).
* ``petition_nlrb_all2024.csv`` - file containing all data for petitions filed between Jan 2015 and Dec 2024 and reported on the NLRB website (note that also decertification elections are included).
* ``overall_nlrb_all2024.csv`` - file containing petitions and elections merged data (note that also decertification elections are included, they will be excluded when the event dictionary is created in ``../process-data/create_events_dict_fb.ipynb``).
* ``mapping_fb_unions.csv`` - file containing the mapping between union names and facebook accounts.
* ``full_set_frames.csv`` - sample of the FB data annotated by coders for discourse frames labels. Note that there is a "political endorsement" frame included here as it was originally of interest for the classification. The folder also contains ``train_set_framing.csv``and ``test_set_framing.csv`` which are the train and test set splits of the data, respectively.
* ``fb_data_with_predictions.csv`` - full FB data annoatated by the multi-label classifier for the presence of discourse frames and the predicted probabilities for the presence of emotion in posts.
* ``topic_info_all.csv`` - results from applying topic modeling with BERTopic on all FB data. This file is shared for transparency and contains all retrieved topic, while only the first 10 are reported in the paper.
* ``union_info.xlsx`` - information about unions present in the data, including traditional union-level controls and social media activity data.
* ``right-to-work_states.csv``- list of U.S. states in which the right-to-work law is in place.