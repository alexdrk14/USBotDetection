# Machine learning Bot detection technique, based on United States election dataset (2020).

Code folder contain all scripts for:
- feature selection 
- parameter fine tuning
- final model performance evaluation

Developed Bot Detection ML framework was trained and tested over Twitter dataset.
The particullar dataset was collected with use of Twitter API during the 2020 US Election period. During this period we manage to collect tweets/retweets that was containing particullar hashtags that is corelated with 2020 US Election topic.
We provide our dataset in form of two separated files (). In those files we provide already extracted features for each user that was labeled according to our labeling process.
The 'us_2020_election_data.csv' contain user features and labels that was active in twitter during the month of September, we utilize it during our training/validation/testing phase of our implemntation. The second file 'us_2020_election_data_october.csv' is utilized in order to identify how developed model can perform over the onseen data that was collected during the month of October.
Both files was labeled with use of Botometer and BotSentinel Bot Detection tools. We keep only union of users where both tools agree if particular account is bot or normal user. Such labeling method allow us to keep user labels cleer as possible.
We remove user ids in order to keep private the user identity, we provide only labels and extracted features which do not provide any sensitive user information.

In plot folder we store performance metrics result in form of figures. 


Execution flow:
- Step 1: code/feature_selection.py
- Step 2: code/model_fine_tuning.py
- Step 3: performance_measurement.py

Developed framework provides state of the art performance over the Twitter US Election data. Based on the performance of unseen data we assume that is possible utilization of developed method over data more that one moth after of training data, since the performance drop is not significant.


 
