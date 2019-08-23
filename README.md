# Sample-Python-Program
Python Program for Decision Tree

Decision Tree Implementation Details:

Python Version: 3.7.4

Libraries Used: Pandas, Numpy

Explanation:

decisiontree.py constructs a decision tree with the parameters described below -
max_depth: the maximum depth upto which the decision tree can be built
           values allowed: any integer
split_val_metric: the metric used to divide the data into two parts for a feature
           values allowed: 'mean' or 'median'
min_info_gain: the minimum information gain required to make the split
           values allowed: any real number
split_node_criterion: the criterion used to measure the information gain for the node
           values allowed: 'gini' or 'entropy'

Training  and Testing data are to be provided as a pandas DataFrame
Training and Testing lables are to be provided as a numpy array

Sample code to run decisiontree.py:
The given code will construct a decision tree of max_depth=2 from the given data
    X_train=pd.DataFrame(np.array([[0,0],[1,1],[0,1],[1,0]]))
    Y_train=np.array([1,1,0,0])
    dec= DecisionTree(max_depth=2)
    dec.train(X_train,Y_train)
    print(dec.predict(pd.DataFrame(np.array([[0,0]]))))



