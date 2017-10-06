
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

pd.set_option('display.max_columns', 100) # to display all columns at all time
pd.options.mode.chained_assignment = None # to ignore false-positive warnings about chained assignments


# In[2]:

data = pd.read_csv('ElectionsData.csv', header=0)
data_copy = data.copy() # clean copy of the data, will be kept wihout any manipulation


# Let's see what we're working with:

# In[3]:

data.head(10)


# In[4]:

data.info()


# # Attribute Types
# 
# 
# Let's see what we're dealing with:

# In[5]:

def count_categories(attr):
    return len(data[attr].astype('category').cat.categories)

obj_attr = [(col, count_categories(col))  for col in data if data[col].dtype==np.object]
obj_attr_names = map(lambda x: x[0], obj_attr)
for attr, cnt in obj_attr:
    print "%-30s %5d %s" % (attr,cnt, "BINARY" if cnt==2 else "")


# Great! out of 10K rows, all attributes have a very small amount of distinct values,
# meaning they are ALL categorical, and 6 of them are binary.
# 
# We'll transform them to __categorical__, for now.

# In[6]:

for attr, cnt in obj_attr:
        data[attr] = data[attr].astype('category')

data.info()


# # Imputation
# As we can see from data.info(), all attributes have missing values.
# The most naive thing to do is simply remove the rows with the missing values. This, of course, will make us lose data.
# One advantage to this though is that it guarantees that all data is legitimate and not inferred in any way.
# 
# Let's see how many data rows we'd lose that way:

# In[7]:

def get_nan_per_row_counter():
    return Counter(data.isnull().sum(axis=1).tolist())

def plot_pie_nan_per_row():
    counter = get_nan_per_row_counter()
    labels, histogram = zip(*counter.most_common())
    fig1, ax1 = plt.subplots()
    ax1.pie(histogram, labels=labels,
            colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral'],
            explode = [0.1] * len(histogram),
            autopct = lambda(p): '{:.0f}  ({:.2f}%)'.format(p * sum(histogram) / 100, p)
           )
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


# In[8]:

plot_pie_nan_per_row()


# As the pie-chart demonstrates, we'd lose almost 20% of our data, which is more than we can afford.
# 
# we can also see the vast majority of rows have almost-complete data, meaning that keeping them all and filling out all missing values will maintain a high per-row credibility.
# 
# So that is what we will do! (This is also what we have to do, because this is what the assignment says...)
# We now have to impute every single missing data-piece, using the methods we learned in class.

# ## Class-Restricted observations:
# 
# We found some strong connections between the "Vote" and some of the categorical attributes.
# This is great because the "Vote" is present in all rows.
# 
# Let's observe the distribution of different attributes, with respect to __Vote__:

# In[9]:

def plot_crosstab(attr1, attr2):
    a1_to_a2 = data[[attr1,attr2]].groupby([attr1, attr2]).size().unstack().fillna(0)
    colors = [col[:-1] for col in a1_to_a2.columns.values] if attr2 == 'Vote' else None
    ax = a1_to_a2.plot(kind='bar', stacked=True, color=colors)
    ax.set_xlabel(attr1)
    ax.set_ylabel(attr2)
    ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5,1.3))

def plot_attr_to_vote(attr):
    plot_crosstab(attr, 'Vote')


for attr, cnt in obj_attr:
    if attr=='Vote':
        continue
    plot_attr_to_vote(attr)
    
plt.show()


# We can clearly see that given the Vote, we can with almost certainty guess the values of the following attrbiutes:
# * Looking_at_poles_results
# * Will_vote_only_large_party
# * Financial_agenda_matters
# 
# Another abnormal observation is made on 'Most_Important_Issue',
# We see these relationships:
# 
# If someone votes to one of these __Parties__ we can see these are the most important issues for them are:
# * Greens --> (Foreign_Affairs OR Other)
# * Pinks  --> (Military OR Other)
# * Whites --> (Foreign_Affairs OR Military)
# 
# If someone's most important __Issues__ are one of these, then we can see that they've voted to one of these parties:
# * Foreign_Affairs --> (Greens OR Whites)
# * Military       --> (Pinks OR Whites)
# * Other        --> (Greens OR Pinks)
# 
# there's a potential multi-attribute dependency here, we will try to find it later on
# 
# 
# Let's fill in the missing values in the 3 above attributes based on these relations:

# In[10]:

vote_to_cats = data[['Vote','Looking_at_poles_results','Will_vote_only_large_party','Financial_agenda_matters']].dropna().drop_duplicates().set_index(['Vote']).to_dict('index')
vote_to_cats
for party, cat_dict in vote_to_cats.iteritems():
    data[data.Vote==party] = data[data.Vote==party].fillna(cat_dict)

plot_pie_nan_per_row()


# Plot every category-category relation:

# In[11]:

#for c1,c2 in combinations(obj_attr_names,2):
#    if c1=='Vote' or c2=='Vote':
#        continue
#    plot_crosstab(c1, c2)
    
#plt.show()


# We could not infer anything useful from it, and have therefore left the code commented out.

# ## Pearson's correlation coefficient
# 
# We will find the coefficient of the linear correlation between every pair of attributes. The pairs with the highest correlation coefficient will be used to impute each other.

# In[12]:

pearson = data.corr(method='pearson')


# In[13]:

def get_most_correlated(corr_mat, col, thresh):
    res = pearson[col].reindex(pearson[col].abs().sort_values(ascending=False).index)
    res = res[res.index != col]
    return res[res >= thresh]


pearson_coeff_thresh = 0.95

# a dict mapping each col with its 3 most-correlated cols, only if the corr with the best of them is >=0.9 (in abs value)
sorted_pearson_per_col = {col:                           get_most_correlated(pearson, col, pearson_coeff_thresh) for col in pearson                         if get_most_correlated(pearson, col, pearson_coeff_thresh).any()}

# for example, here are the columns that are most correlated wth 'Avg_monthly_expense_when_under_age_21'
sorted_pearson_per_col['Avg_monthly_expense_when_under_age_21']


# Look at that! we have a PERFECT correlation between Avg_Residancy_Altitude and Avg_monthly_expense_when_under_age_21!
# 
# This means that there's an exact linear relationship between the two. let's check it out:

# In[14]:

(data['Avg_monthly_expense_when_under_age_21'] / data['Avg_Residancy_Altitude']).head(5)


# as we can see, Avg_monthly_expense_when_under_age_21 = 3 * Avg_Residancy_Altitude for every row that contains them both,
# i.e. every row that is missing only one of them, can be completed precisely.
# 
# This method seems valueable, we shall continue with it for other attributes, who have very high linear correlation with each other.
# 
# What we'll do is for each row, for each NaN value in one of those attributes, if there's data in one of its highly-correlated attributes, take this data, multiplied by the mean ratio.

# In[15]:

# commented-out on purpose - just shows us the content of the dict for sanity check
#for col in sorted_pearson_per_col:
#    print sorted_pearson_per_col[col],"\n\n"


# In[16]:

for col, top_corr_cols in sorted_pearson_per_col.iteritems():
    for corr_col in top_corr_cols.index:
        ratio = (data_copy[col] / data_copy[corr_col]).mean()
        cols = data_copy[[col, corr_col]]
        cols[corr_col] = cols[corr_col].map(lambda x: x*ratio)
        data[col].fillna(cols[corr_col], inplace=True)

data


# In[17]:

plot_pie_nan_per_row()


# we now have ~550 more complete rows!
# and about ~110 rows which had more than a signle missing value now have one or less.
# only by using columns that have a pearson's correlation coefficient above 0.95 (in abs value), meaning this is a pretty close approximation.

# ## Spearman's rank
# 
# Again we will find the correlation between every pair of attributes. The pairs with the highest correlation coefficient will be used to impute each other.
# 
# A high Spearman Rank between X and Y, means that if for two given samples s1 and s2, if x1 > x2 we shold expect y1 > y2.
# This implies that the preferred method of imputation based on this rank, is by interpolating Y as a function of X (and vice-versa).

# In[18]:

spearman = data.corr(method='spearman')


# In[19]:

spearman_coeff_thresh = 0.95

# a dict mapping each col with its 3 most-correlated cols, only if the corr with the best of them is >=0.9 (in abs value)
sorted_spearman_per_col = {col:                           get_most_correlated(pearson, col, pearson_coeff_thresh) for col in pearson                         if get_most_correlated(pearson, col, pearson_coeff_thresh).any()}


# In[20]:

#looking at the dict just for sanity check - commented out
sorted_spearman_per_col


# In[21]:

from scipy import interpolate

## we copy our data so when we impute, we only use original data and not newly imputed data.
## this prevents propagation of approximation errors.
data_copy = data.copy()

for col, top_corr_cols in sorted_pearson_per_col.iteritems():
    for corr_col in top_corr_cols.index:
        print col, "<==>", corr_col
        cols = data_copy[[col, corr_col]].sort_values(by=corr_col)
        print "after sort"
        x_valid = np.isfinite(cols[corr_col])
        y_valid = np.isfinite(cols[col])
        both_valid = x_valid & y_valid
        y_missing_x_valid = x_valid & ~y_valid
        func = interpolate.interp1d(cols[corr_col][both_valid], cols[col][both_valid])
        print "after fit"
        data[col][y_missing_x_valid] = func(cols[corr_col][y_missing_x_valid])
        print "after predict"
        print "\n"

data


# In[22]:

plot_pie_nan_per_row()


# # The Complete Completion
# 
# We now tried to find a method to properly impute all other missing data, while taking into account mostly the preservation of 
# relations between features, to best fit our mission of applying machine-learning on the data.
# 
# we considered EM as taught in the lectures, but found the assumption of Multivariate-Gaussian distribution to mostly incorrect when it comes to our data.
# 
# after looking for a while, and considering different methods, we have decided to use SoftImpute, based on this paper: http://web.stanford.edu/~hastie/Papers/mazumder10a.pdf.
# 
# "SoftImpute" also conviniently has an open-source implementation in Python.

# In[23]:

from fancyimpute import SoftImpute

data_no_cat = data.drop(obj_attr_names, axis=1)

completed = SoftImpute().complete(data_no_cat)

data[data_no_cat.columns] = completed

plot_pie_nan_per_row()


# We still have to deal with categorical attributes,
# we will naively impute them with a class-restricted most-common approach:

# In[24]:

from sklearn.preprocessing import Imputer

news = []

for vote in data.Vote.unique():
    vote_restricted = data[data.Vote == vote]
    for c in obj_attr_names:
        if c == 'Vote':
            continue
        print vote,"+",c,":",
        most_common = pd.get_dummies(vote_restricted[c]).sum().sort_values(ascending=False).index[0]
        print most_common
        vote_restricted[c].fillna(most_common, inplace=True)
    news.append(vote_restricted)

data = pd.concat(news)


# In[25]:

plot_pie_nan_per_row()


# In[26]:

data = data.sort_index()


# # Type/Value modification

# Now we want to re-assign values to categorical and binary attributes:
# 
# The categorical, non-binary features will be split into one-hot vectors.
# The "binary" ones will be assigned 0 and 1.
# 
# __BUT:__ notice Age_group has somewhat "ordinal" values, ['Below_30', '30-45', '45_and_up'], so we would like to map these to integers with the correct ordering.

# In[27]:

data['Gender_Int'] = data['Gender'].map({'Female':0, 'Male':1}).astype(float)
data['Voting_Time_Int'] = data['Voting_Time'].map({'By_16:00':0, 'After_16:00':1}).astype(float)

data = data.drop(['Gender','Voting_Time'],axis=1)

for attr in ['Married','Looking_at_poles_results','Financial_agenda_matters','Will_vote_only_large_party']:
    data[attr+'_Int'] = data[attr].map({'No':0, 'Yes':1}).astype(float)
    data = data.drop([attr],axis=1)

# Handle categorical columns and add one-hot vectors
for attr in ['Most_Important_Issue','Main_transportation','Occupation']:
    data = pd.concat([data, pd.get_dummies(data[attr],prefix=attr)], axis=1)
    data = data.drop([attr],axis=1)

# For convenience, we want 'Vote_Int' to be at the beginning, but we're not dropping 'Vote' just yet
data = pd.concat([pd.get_dummies(data['Vote'],prefix='Vote'),data], axis=1)

#for attr,cnt in obj_attr:
#    if attr=='Vote':
#        data[attr] = data[attr].astype('category').cat.rename_categories(range(1,cnt+1)).astype('float')

data['Age_group_Int'] = data['Age_group'].map({'Below_30':0, '30-45':1, '45_and_up':2}).astype(float)
data = data.drop(['Age_group'],axis=1)


# In[28]:

data.head(10)


# In[29]:

data.info()


# # Outlier Detection
# 
# We will use Isolation Forests for this task.
# We've decided to go with Isolation Forests as they are based on Random Forests and therefore deal well with multidimensional data, like in our case.

# In[30]:

from sklearn.ensemble import IsolationForest

df = data.drop(['Vote'],axis=1)
outliers_fraction = 0.05

clf = IsolationForest(n_estimators=300, max_samples=0.5, contamination=outliers_fraction, n_jobs=-1)

clf.fit(df)
y_pred = clf.predict(df)
outliers = np.where(y_pred==-1)[0]


# In[31]:

outliers


# ## Test Classification before/after removing the outliers

# In[32]:

from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

vote_list = [v for v in data.columns.values if 'Vote' in v and v != 'Vote']
df = data.drop(vote_list,axis=1)
for attr,cnt in obj_attr:
    if attr == 'Vote':
        df[attr] = df[attr].astype('category').cat.rename_categories(range(1,cnt+1)).astype('float')


# In[33]:

df


# In[34]:

avg_tree = 0
#avg_svm = 0
    
train_data_X = df.drop(['Vote'],axis=1).values
train_data_Y = df['Vote'].values

# Prepare train and test data using cross validation
X_train, X_test, Y_train, Y_test = train_test_split(train_data_X,train_data_Y)
   
for i in range(10):
 
    # Example usage 1
    forest = RandomForestClassifier(n_estimators = 10)
    clf = forest.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    avg_tree += metrics.accuracy_score(Y_test, Y_pred)

    #SVM performs badly because we have not selected a feature subset yet
    
    # Example usage 2
    #svm = SVC()
    #clf = svm.fit(X_train, Y_train)
    #Y_pred = clf.predict(X_test)
    #avg_svm += metrics.accuracy_score(Y_test, Y_pred)
    
print avg_tree/10 #, avg_svm/10


# ### After removing...

# In[35]:

df_no_outliers = df.drop(data.index[outliers])

df_train_data_X = df_no_outliers.drop(['Vote'], axis=1).values
df_train_data_Y = df_no_outliers.Vote.values


# In[36]:

avg_tree = 0
#avg_svm = 0

# Prepare train and test data using cross validation
df_X_train, df_X_test, df_Y_train, df_Y_test = train_test_split(df_train_data_X, df_train_data_Y)

for i in range(10):
    
    # Example usage 1
    forest = RandomForestClassifier(n_estimators = 10)
    clf = forest.fit(df_X_train, df_Y_train)
    df_Y_pred = clf.predict(df_X_test)
    avg_tree += metrics.accuracy_score(df_Y_test, df_Y_pred)

    #SVM performs badly because we have not selected a feature subset yet
    
    # Example usage 2
    #svm = SVC()
    #clf = svm.fit(df_X_train, df_Y_train)
    #df_Y_pred = clf.predict(df_X_test)
    #avg_svm += metrics.accuracy_score(df_Y_test, df_Y_pred)

print avg_tree/10  #, avg_svm/10


# Removing the outliers didn't seem to improve our accuracy, meaning the outliers did not introduce a strong bias in the first place

# ## Normalization
# 
# We want to scale the data to be of more use in certain classifiers, like those based on euclidean/L2 distance, such as KNN.
# 
# We will leave one-hot vectors and binary attributes as they are, and scale other attributes to be with a variance of 1 and a mean of 0.

# In[37]:

from sklearn.preprocessing import scale


# In[38]:

attrs = [attr for attr in data if count_categories(attr) > 2 and attr != 'Vote']

for attr in attrs:
    data[attr] = scale(data[attr])


# In[39]:

data


# ## DF - what is this DataFrame below?
# 
# The following dataframe is the same as the normal "data" DataFrame, except that the one-hot vector for Votes has been removed, and the 'Vote' attribute values have been replaced with numbers. This is because train_test_split does not work with String based attributes.
# 
# This DataFrame serves for testing things and selecting the feature subset.

# In[40]:

df = data.drop(vote_list,axis=1)
for attr,cnt in obj_attr:
    if attr == 'Vote':
        df[attr] = df[attr].astype('category').cat.rename_categories(range(1,cnt+1)).astype('float')


# In[41]:

#import scipy.stats as stats
#for col in [c for c in data if data[c].dtype == np.float64]:
#    ax = data[col].plot(kind='density')
#    ax.set_xlabel(col)
#    plt.show()


# # Splitting the dataset

# In[42]:

from sklearn.model_selection import train_test_split
data_X = data.drop(vote_list+['Vote'], axis=1)
data_Y = df['Vote']

#Split data after preprocessing

# split first level, having train with 60% of data.
x_train, other_X, y_train, other_Y = train_test_split(data_X,data_Y,train_size=.6, stratify=data_Y, random_state=313341596)

# split scecond level, operating on the remaining 40%, splitting in half.
x_test, x_validation, y_test, y_validation = train_test_split(other_X,other_Y,train_size=.5, stratify=other_Y, random_state=302325709)

# "Merge" the X and Y of each set (will save in a single CSV file)

afterpp_train = data.iloc[x_train.index]
afterpp_test = data.iloc[x_test.index]
afterpp_validation = data.iloc[x_validation.index]

#Split original data based on this split

orig_train = data_copy.iloc[x_train.index]
orig_test = data_copy.iloc[x_test.index]
orig_validation = data_copy.iloc[x_validation.index]




afterpp_train.to_csv('afterpp_train.csv')
afterpp_test.to_csv('afterpp_test.csv')
afterpp_validation.to_csv('afterpp_validation.csv')

orig_train.to_csv('orig_train.csv')
orig_test.to_csv('orig_test.csv')
orig_validation.to_csv('orig_validation.csv')


# # Bonuses
# 
# TL;DR: We've implemented:
# 
# * Hybrid Selection Scheme (see BDS, which isn't the actual BDS algorithm but a variation of it. more explanation there).
# * Relief algorithm.
# * Role of attributes with respect to classes ('Vote') as seen in the Imputing section.
# * SFS implementation.

# ## Relief Algorithm
# 
# This is the implementation of the relief algorithm.
# 
# ## DISCUSS ADVANTAGES AND DISADVANTAGES OF RELIEF
# 
# The clear advantage of the Relief algorithm [family] is that they are able to detect conditional dependencies between attributes. Other algorithms can also do this but mostly indirectly and as a result of other factors. (For example, in SFS, some attributes aren't as useful once other attributes have been added because they may have much in common, but the other feature may be just as useful.). In Relief, we first go over all the features, find the best ones, and then pick those at once.
# 
# A clear disadvantage is that Relief takes a long time to run.
# We also noticed it produces feature-subsets much larger (2x~) than subsets produced by other methods we have tried, which means larger runtime complexity when fitting and predicting.
# 
# ## WERE THERE ANY FEATURES ALWAYS / NEVER SELECTED BY IT?
# 
# We've noticed that some features are never selected, and others always are.
# We couldn't spot any "bias" in the algorithm itself (as we did in SFS - will be explained later), but it does seem to align with the results of other algorithms as to which features are most important.
# 

# In[43]:

def find_closest(data,index,row):
    nearhit_dist = nearmiss_dist = None
    nearhit = nearmiss = None

    for idx, cur_row in data.iterrows():
        if idx == index:
            continue
        cur_vote = cur_row.Vote
        dist = sum([(row[c]-cur_row[c])**2 for c in data if c != 'Vote'])
        if cur_vote == row.Vote:
            if nearhit_dist is None or dist < nearhit_dist:
                nearhit_dist = dist
                nearhit = cur_row
        else:
            if nearmiss_dist is None or dist < nearmiss_dist:
                nearmiss_dist = dist
                nearmiss = cur_row
    return nearhit, nearmiss
        

def relief(data, samples=0.2,tau=0):
    weights = {}
    #initialize the weights
    for f in data.columns.values:
        if 'Vote' in f:
            continue
        weights[f] = 0
    #go over the samples
    i = 0
    for index, row in data.sample(frac=samples).iterrows():
        i = i + 1
        print "i =", i
        vote = row.Vote
        #find nearest from class, and its index
        #find nearest from outside class
        nearhit, nearmiss = find_closest(data,index,row)
        for f in data.columns.values:
            if 'Vote' in f:
                continue
            #print f
            #weights[f] = weights[f] + (xi-nearmiss(xi))^2 - (xi-nearhit(xi))^2
            weights[f] = weights[f] + (row[f] - nearmiss[f])**2 - (row[f] - nearhit[f])**2
        attrs = [attr for attr, w in weights.iteritems() if w>tau]
        print "attrs_size:",len(attrs)
    return attrs
    
#attrs = relief(data,samples=0.01)
#attrs


# In[44]:

print len(attrs)
open_set = set(attrs)


# In[45]:

#df


# ## SFS
# 
# Below is the implementation of the SFS Feature Selection Algorithm.
# 
# 
# ## DISCUSS ADVANTAGES AND DISADVANTAGES OF RELIEF
# 
# * SFS is prone to Local Maxima and might not include useful features. This is basically an SAHC algorithm. This could be solved, or at least improved, with Random Restarts, Side Stepping, and Simulated Annealing.
# * The first disadvantage we can think of is the fact that SFS is not very conservative. This means that there might be useful features it will not include, because it will reach a local maximum.
# * SFS is affected by bias of the wrapped classifier, and somewhat "inherits" its disadvantages. One example would be RandomForest almost always selecting "Last_School_Grades" as its first attribute, because it improves its accuracy the most. A different classifier might get more use out of a different feature and select it instead.
# * SFS might miss out on multi-attribute relations due to its nature of starting with an empty subset and greedily expanding it and stops right a the point that another single step doesn't benfit us (even though perhaps two more steps could overcome the local maximum).
# 
# ## WERE THERE ANY FEATURES ALWAYS / NEVER SELECTED BY IT?
# 
# We've noticed that some features are never selected, and others always are.
# It mostly occurs on the first step of the algorithm, which always starts at the same position, and chooses the feature that by itself helps gain the most accuracy, and even though it might not be as useful once other features are selected, SFS will not remove it from the subset.
# 
# As part of our feature selection we've ran SFS multiple times and counted the number of appearances each attribute appeared in one of the subsets returned by SFS. There we can see which features were almost always selected and which features almost never, or were very rarely selected.
# 

# In[46]:

#df


# In[47]:

from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

def _sfs(df,learning_model,S=None,cur_max=0):
    if S is None:
        S = set(['Vote'])
    #print S
    max_feature = None
    for col in [c for c in df if c not in S]:
        S.add(col)
        subset = df[list(S)]
        train_data_X = subset.drop(['Vote'], axis=1).values
        train_data_Y = subset.Vote.values
        # Prepare train and test data using cross validation
        X_train, X_test, Y_train, Y_test = train_test_split(train_data_X, train_data_Y)
        clf = learning_model.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        tmp = metrics.accuracy_score(Y_test, Y_pred)
        if(tmp > cur_max):
            cur_max = tmp
            max_feature = col
        S.remove(col)
    if (max_feature is not None):
        S.add(max_feature)
        #print cur_max
        S, cur_max = _sfs(df,learning_model,S,cur_max)
    return S, cur_max


#df: dataframe, d-dimensional feature-set
#learning_model: classifier by which to measure predictive power, higher = better
#iterations: this is to allow restarts because the learning_model may be random. picks the best subset out of all
def sfs(df,learning_model,iterations=1):
    best_score = 0
    best_subset = set()
    #because the function is susceptible to local maximums,
    #we will run it with random restarts and take the best subset
    for i in range(iterations):
        S, score = _sfs(df,learning_model)
        if(score > best_score):
            best_score = score
            best_subset = S
    if 'Vote' in best_subset:
        best_subset.remove('Vote')
    return best_subset, best_score

# Example usage 1
#forest = RandomForestClassifier(n_estimators = 15)
#S, accuracy = sfs(df,forest,5)
#print S, accuracy

# Example usage 2
#svm = SVC()
#S, accuracy = sfs(df,svm)
#print S, accuracy


# ## Hybrid Feature Selection
# 
# This is a variation of the BDS algorithm.
# As input, it receives:
# * A DataFrame
# * A learning model (basically a classifier or something that can judge the quality of the subset)
# * Number of iterations, if we want to allow restarts because of non-deterministic learning models
# * Subset of features to filter from. If none is specified, start with all features.
# 
# The modified BDS Algorithm runs like a regular BDS algorithm but returns the best subset it has seen in either path for the SFS and SBS, and not just the subset where the SFS and SBS meet at.
# 
# The rationale is that we'll return the best subset from either SBS and SFS and indirectly, also the best subset BDS would return, all combined into one algorithm.
# 
# We've allowed it the feature to begin with a given subset and pick a subset from that, because we will give it the subset found by Relief.

# In[49]:

from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

def _bds(df,learning_model,sfs_open=None):
    if open_set is None:
        sfs_open = set(df.columns.values)
        sfs_open.remove('Vote')
    sbs_open = set.copy(sfs_open)
    
    S_sfs = set()
    S_sbs = set.copy(sfs_open)
    
    S_max = None
    
    cur_max_forward = 0
    cur_max_backward = 0
    cur_max = 0
    
    train_data_Y = df.Vote.values
    
    while S_sfs != S_sbs:
        
        max_feature = None
        min_feature = None
        changed = False
        
        print "#### FORWARD ####"
        print S_sfs
        #forward selection
        #find a feature in sfs_open and add it
        for f in sfs_open:
            S_sfs.add(f)
            subset = df[list(S_sfs)]
            
            train_data_X = subset.values
            
            # Prepare train and test data using cross validation
            X_train, X_test, Y_train, Y_test = train_test_split(train_data_X, train_data_Y)
            clf = learning_model.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            tmp = metrics.accuracy_score(Y_test, Y_pred)
            
            #pick the best feature to add
            if max_feature is None:
                max_feature = f
            if tmp > cur_max_forward:
                cur_max_forward = tmp
                max_feature = f
            if tmp > cur_max:
                cur_max = tmp
                S_max = S_sfs
            S_sfs.remove(f)
        if (max_feature is not None):
            S_sfs.add(max_feature)
            #sbs can't remove feature selected by sfs
            sbs_open.remove(max_feature)
            sfs_open.remove(max_feature)
            
        print "#### BACKWARD ####"
        print S_sbs
        #backward selection
        for f in sbs_open:
            S_sbs.remove(f)
            subset = df[list(S_sbs)]
            
            train_data_X = subset.values
            
            # Prepare train and test data using cross validation
            X_train, X_test, Y_train, Y_test = train_test_split(train_data_X, train_data_Y)
            clf = learning_model.fit(X_train, Y_train)
            Y_pred = clf.predict(X_test)
            tmp = metrics.accuracy_score(Y_test, Y_pred)
            
            #find least damaging feature to remove
            if min_feature is None:
                min_feature = f
            if(tmp > cur_max_backward):
                cur_max_backward = tmp
                min_feature = f
            if tmp > cur_max:
                cur_max = tmp
                S_max = S_sfs
            S_sbs.add(f)
        if (min_feature is not None):
            S_sbs.remove(min_feature)
            sfs_open.remove(min_feature)
            sbs_open.remove(min_feature)
    return S_max, cur_max


#df: dataframe
#learning_model: classifier by which to measure predictive power, higher = better
#iterations: this is to allow restarts because the learning_model may be random. picks the best subset out of all
def bds(df,learning_model,iterations=1,open_set=None):
    best_score = 0
    best_subset = None
    #because the function is susceptible to local maximums,
    #we will run it multiple times and pick the best subset
    for i in range(iterations):
        S, score = _bds(df,learning_model,open_set)
        if(score > best_score):
            best_score = score
            best_subset = S
    return best_subset, best_score

# Example usage 1
#forest = RandomForestClassifier(n_estimators = 15)
#S, accuracy = bds(df,forest,iterations=1,open_set=open_set)
#print S, accuracy

# Example usage 2
#svm = SVC()
#S, accuracy = sfs(df,svm)
#print S, accuracy


# # Select the feature subset
# 
# 1. Run SFS Multiple times.
# 2. Run Relief.
# 3. Run BDS on the subset of features selected by Relief.
# 4. Pick the most common features selected by SFS, united with the BDS run.

# In[50]:

from collections import Counter

sfs_runs = 20
sfs_feature_threshold = sfs_runs*0.2

relief_threshold = 0.01

SFS_subset = set()

forest = RandomForestClassifier(n_estimators = 15)
sfs_counter = Counter()
for i in range(sfs_runs):
    print "############"
    print i
    S, accuracy = sfs(df,forest)
    print S
    print accuracy
    sfs_counter += Counter(S)
    print sfs_counter

most_common = {f for f,c in sfs_counter.most_common() if c > sfs_feature_threshold}

print "### DONE WITH SFS ###"
print most_common

print "### RELIEF ###"
relief_features = set(relief(data,samples=0.01))
print relief_features
print "### DONE WITH RELIEF ###"

print "### BDS ###"

bds_features, accuracy = bds(df,forest,iterations=1,open_set=relief_features)
print "### DONE WITH BDS ###"
print bds_features
print accuracy

feature_subset = most_common | bds_features

print "FEATURE SUBSET:"
print feature_subset


# In[51]:

feature_subset


# In[52]:

len(feature_subset)


# In[ ]:



