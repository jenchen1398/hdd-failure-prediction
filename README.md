# hdd-failure-prediction
Step-by-step tutorial for predicting hard drive failures in large-scale data centers using 
naive bayes algorithms.
## Inspiration
During my summer internship in 2017 at Hudson River Trading (HRT), I was assigned the task of predicting 
hard drive failures for machines. In modern day large-scale data centers, predicting hard drive
failures can be extremely useful to streamline maintenance and backup data before the data is
corrupted by a HDD failure. Thanks to the research at Backblaze, a company that specializes in
cloud & backup storage services, I was able to train my predictive models with large amounts of
data and achieve 99.5%+ accuracy. In this iPython notebook, I use data from Backblaze that I 
preprocessed to include approx. a 1:10 ratio of failed to working drives to achieve 95%+ accuracy
in prediction. (This ipynb achieves lower accuracy because I use less data to train and have a
higher failed:working ratio in my data than my model for HRT did).

