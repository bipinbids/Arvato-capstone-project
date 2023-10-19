# Arvato  Challenge 

## Project Overview

All flourishing business tend to grow in size and revenue by bringing on new customers and the process of doing so can be quite hectic, As time passes humanity as tried its best to get better at things, In this effort, it's interesting to wonder if Machine Learning and Data Analysis could be used to try and reach out to the most promising individuals, in order for the business to better serve them and bring them on as future customers.

Here, we will be using the data provided by Arvato Financial Solutions, who happen to be a Bertelsmann subsidiary, in an attempt to use some Machine learning Algorithms to drive customer accusation for them.

Our attempt at discovering the potential customers will be on three fronts

Part one : - Here, we inspect the relationship between the demographics of the company's existing customers with the general population of Germany and attempt to detect parts of the general population of Germany who are most likely to be a part of the mail-order company's customer base, and who are least likely to do so.

In this direction, we shall use unsupervised learning techniques such as PCA and Kmeans to help us inspect the relation between the above-mentioned categories.

Part two : -Here, data which was provided was used to build a predictive model, each row of the data had information about an individual that was targeted in the previous campaign, individuals who seemed to be promising customers was made a part of the new campaign
In this direction, we shall use supervised learning techniques such as Gradient Boosting Classifier, ADA Boost Classifier, Random Forest Classifier, Bagging Classifier, and Logistic Regression.

Part three :- The optimized and tuned model from the previous part is deployed against the test dataset the results are used to participate in a  compitation.


## Files 
- final_project_submission.ipynb
	-A jupyter notebook that performs all the tasks described in the project overview and produces a csv named final_submission.csv.
- manually_created_csv.csv	
	-This file was manually created ,this file consist feature name, missing information for each type that was seen, along with its type
- workspace_utils.py
	-This file was provided by the tech support of udacity and was used to execute a long running code without shutting down the workspace
- README.md
	-Read this first to have a brief idea about the project.

## Most Helpful Libraries Used
- [Sklearn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

## What Do The Results Say?

most of the effort in this project was put towards data preparation before further steps could be applied, CRISP-DM Process had to be applied to both parts of the project. The dataset provided was imbalanced, this led us down a path of cautious steps while choosing the classifier to work with, lets once again recall our journey.

Part one :- The main bulk of analysis was seen in part 1, we used PCA to reduce the noise so that the clustering algorithm (KMeans) could perform better. The KMeans Algorithm further distinguished the groups of individuals who formed the ideal customer base for the company, we then used LogisticRegression to figure out the most important features for each of these ideal groups.

Part two :- In part 2 we built various supervised learning classifiers (LogisticRegression, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, LinearRegression) trained it against the training dataset to find that GradientBoostingClassifier had outperformed other classifiers so, we further tuned the GradientBoostingClassifier using gridsearchcv, then we took a look at the what model has to say the most important features of the dataset was. Last but not least the optimized model was deployed against the test dataset to figure out the company's ideal customer base.

ROC AUC was used as a measure to evaluate the performance of the classifiers instead of accuracy.

gradient boosting clearly outperforms its competitors. One of the main reasons for this behavior might be because gradient boost has a built-in approach for handling class inbalance and this greatly helped in our case, themodel further under went an optimization using gridsearch.

The top five most important features according to the trained and optmised model are
1) D19_SOZIALES
2) ANZ_KINDER
3) D19_BANKEN_LOKAL
4) D19_GARTEN	
5) GEBURTSJAHR

## REFERENCES
1) [How to find optimal number of clusters](https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f)

2) [Get feature importance from GridSearchCV](https://stackoverflow.com/questions/48377296/get-feature-importance-from-gridsearchcv)

3) [Grid search for model tuning ](https://towardsdatascience.com/grid-search-for-model-tuning-3319b259367e)

4) [The art and science of dealing with imbalanced datasets](https://medium.com/@humansforai/the-art-and-science-of-dealing-with-imbalanced-datasets-209b448a11c5)

5) [Python Seaborn Tutorial For Beginners](https://www.datacamp.com/community/tutorials/seaborn-python-tutorial?utm_source=adwords_ppc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=278443377095&utm_targetid=aud-390929969673:dsa-473406580275&utm_loc_interest_ms=&utm_loc_physical_ms=9062077&gclid=CjwKCAjwtqj2BRBYEiwAqfzur1C6NlUn5QCR0AwtfJqWlmyqLcsloZYT3W6YTFwMeFFOwZUODRzhERoCcbQQAvD_BwE)

Find a blog describing the project in further detail [here](https://medium.com/@bipinbiddappa/can-meaningful-data-be-used-to-drive-customer-accusation-3c1caa89b352)
