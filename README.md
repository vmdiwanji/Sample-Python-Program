# Sample-Python-Program
<b>Python Program for Decision Tree</b>

<b>Decision Tree Implementation Details:</b>

<b>Python Version:</b> 3.7.4

<b>Libraries Used:</b> Pandas, Numpy

<b>Explanation:</b> <br />

<b><i>decisiontree.py</i></b> constructs a decision tree with the parameters described below - <br />
<br />
<b>max_depth</b>: the maximum depth upto which the decision tree can be built <br />
           values allowed: any integer <br />
<b>split_val_metric</b>: the metric used to divide the data into two parts for a feature <br />
           values allowed: 'mean' or 'median' <br />
<b>min_info_gain</b>: the minimum information gain required to make the split <br />
           values allowed: any real number <br />
<b>split_node_criterion</b>: the criterion used to measure the information gain for the node <br />
           values allowed: 'gini' or 'entropy' <br />

<i>Training  and Testing data are to be provided as a pandas DataFrame </i><br />
<i>Training and Testing lables are to be provided as a numpy array </i><br />

Sample code to run decisiontree.py: <br />
The given code will construct a decision tree of max_depth=2 from the given data
    X_train=pd.DataFrame(np.array([[0,0],[1,1],[0,1],[1,0]])) <br />
    Y_train=np.array([1,1,0,0]) <br />
    dec= DecisionTree(max_depth=2) <br />
    dec.train(X_train,Y_train) <br />
    print(dec.predict(pd.DataFrame(np.array([[0,0]])))) <br />



