This folder contains code to process NLRB union data, facebook data, create a mapping and producing a json file with main unions and events of involvement. 
While by-product files will not be shared (but can be requested to the authors), the final json file will be shared.

# Process NLRB union data

After having collected all necessary offline data (reports of elections) from the NLRB web portal, we need to merge, clean and pre-process the data. Note that, in this project, we will only focus on events related to **reports of elections**. The code can be easily adapted to the other events.

## Obtaining a single merged file with all offline events

Clean and merge the elections data with ``./process_national_reports.ipynb``.

# Get online union activity data 

We first need to create and iteratively clean the hierarchy of unions (affiliation to AFL-CIO, main union, sub-union, district and local levels).
We start from ``./generate_first_hirerarchy.ipynb`` to generate a first version of the hierarchy file. We then use ``./define_hierarchy_info.ipynb`` to keep iterating the cleaning and involve information on AFL-CIO unions and finally ``./clean_hierarchy.ipynb`` to perform additional manual cleaning.

We then run ``./get_main_union_names.ipynb`` to get the file with main union names for AFL-CIO and independent unions.

Then, we can identify the Facebook names of unions using Bing API: 
``./search_data_highlevel.py`` for the "main" union names.

Using Facebook Research Platform (META API) we can then get the data for the accounts (see ``./get_producers_list.ipynb``) to get the producer list of such accounts.
Finally, we retrieve data from FB (this is stored as ``./data/fb_data_with_predictions.csv`` and we already added the information on predicted frames, merging the dataset from FB with predicted frames inferred in ``../classification/``). 
# Generation of framing usage time-series

First, using ``./exploratory_analysis/create_events_dict_fb.ipynb``, we generate a dictionary of events (``./data/events_dict_fb_complete_all_contracts_all2024_new.json``) with the following stucture:
```json
{ "main_union": {
        "handle": "fb_handle",
        "cases": [
            {
                "union": "union name according to offline data",
                "case_number": "case number",
                "case_winning_election_date": "date of election winning for the case" or "None",
                "case_losing_election_date": "date of election losing for the case" or "None",
            },
            ...]},
  ...
}
```
