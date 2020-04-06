# Have to pip3 install <packages> on local machine before import.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups

# Downloads and loads training set data ~/scikit_learn_data/20news_home
newsgroups_train = fetch_20newsgroups(subset='train')

# .shape() prints the size/dimensions of the array
newsgroups_train.filenames.shape

# .target is the integer; corresponds to a news-category
# { 0:'alt.atheism', 1:'comp.graphics' ... 19:'talk.religion.misc'}
category_dict = dict(zip(range(20),list(newsgroups_train.target_names)))

# Print Histogram
data = np.array(newsgroups_train.target)

# Center data around integers
d = np.diff(np.unique(data)).min()
left_of_first_bin = data.min() - float(d)/2
right_of_last_bin = data.max() + float(d)/2
bins = np.arange(left_of_first_bin, right_of_last_bin + d, d)

# Plot Labels and Formatting
# Maps .target to .target_names (indexes 0-19)
plt.xticks(range(20),list(newsgroups_train.target_names), rotation='vertical')
plt.title("Articles per News Category")
plt.xlabel("News Categories")
plt.ylabel("Number of Articles")
plt.hist(data, bins, align='mid', rwidth=0.75)
plt.show()
