# Sample-Python-Program
Python Program for Decision Tree

Decision Tree Implementation Details:

Python Version: 3.7.4

Libraries Used: Pandas, Numpy

Explanation: <br />

decisiontree.py constructs a decision tree with the parameters described below - <br />
<Tab>max_depth: the maximum depth upto which the decision tree can be built <br />
<tab>           values allowed: any integer <br />
<tab>split_val_metric: the metric used to divide the data into two parts for a feature <br />
           values allowed: 'mean' or 'median' <br />
min_info_gain: the minimum information gain required to make the split <br />
           values allowed: any real number <br />
split_node_criterion: the criterion used to measure the information gain for the node <br />
           values allowed: 'gini' or 'entropy' <br />

Training  and Testing data are to be provided as a pandas DataFrame <br />
Training and Testing lables are to be provided as a numpy array <br />

Sample code to run decisiontree.py: <br />
The given code will construct a decision tree of max_depth=2 from the given data
    X_train=pd.DataFrame(np.array([[0,0],[1,1],[0,1],[1,0]])) <br />
    Y_train=np.array([1,1,0,0]) <br />
    dec= DecisionTree(max_depth=2) <br />
    dec.train(X_train,Y_train) <br />
    print(dec.predict(pd.DataFrame(np.array([[0,0]])))) <br />



