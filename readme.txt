Training set : all_workouts_train_condensed.gz
Validation set : all_workouts_validation_condensed.gz
Test set :  all_workouts_TEST.gz (not condensed yet)

Dependencies
-----------

simplejson
numpy
matplotlib

Data processing pipeline
------------------------
1. Run sql_to_json_parser.py on original file EndoMondoWorkouts.sql.gz to get about 10 million workouts (1 workout per line) in all_workouts.gz
2. Run shuffle_and_split_data.py on all_workouts.gz to split data into 2 parts : 50 % for training / validation (all_workouts_train_and_val.gz)  and 50 % for testing (all_workouts_TEST.gz)
3. Condense and clean all_workouts_train_and_val.gz to get all_workouts_train_and_val_condensed.gz
4. Split all_workouts_train_and_val_condensed.gz into 2 parts: 60 % (i.e 30 % of total) for training and 40 % (i.e 20 % of total) for validationto get all_workouts_train_condensed.gz and all_workouts_validation_condensed.gz


Info about files
------------------

EndoMondoWorkouts.sql.gz - 
This is the full data set dump. 
Number of lines : 1267340 
Number of workouts : 10320194
Size of .sql.gz file : 204 GB
Size of .gz file created by sql_to_json_parser : 31 GB

all_workouts.gz- 
This is the output of the sql to json parser on EndoMondoWorkouts.sql.gz. Size is about 31 GB

all_workouts_train_and_val.gz and all_workouts_TEST.gz
Output of shuffling and splitting all_workouts.gz into 2 equal parts - one for training / validation and one for testing

all_workouts_train_and_val_condensed.gz -
Output of running condense_and_clean_data on all_workouts_train_and_val.gz

all_workouts_train_condensed.gz -
This is the TRAINING data : 0.6 fraction of all_workouts_train_and_val_condensed.gz (obtained by running shuffle_and_split on all_workouts_train_and_val_condensed.gz with shuffling disabled) i.e about 30 % of total data.

all_workouts_validation_condensed.gz
This is the VALIDATION data : remaining 0.4 fraction of all_workouts_train_and_val_condensed.gz i.e. about 20 % of total data

all_workouts_TEST.gz - TESTING data (needs to be condensed before using) .. about 50 % of the data

endoMondo5000.sql.gz -
First 5000 lines of the full data set
Number of lines : 5000
Number of workouts : 
Size of .sql.gz file : 
Size of .gz file created by sql_to_json_parser :

endomondo1.txt.gz - 
DO NOT use this file
This is the output of a prematurely ended run of sql_to_json_parser. 
Number of workouts : 5590000 (approximately)
Time taken by old visualize script (which worked on full trace data) : 1678.99035597 seconds
