from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from stability_selection import StabilitySelection
import matplotlib.pyplot as plt


# First stability selection method



Z_stab1=[]
numFeatSelected_stab1 = []
selected_scores1= []

for i in range(num_bootstrap):
    Z_stab1.append([])
    numFeatSelected_stab1.append([])
    selected_scores1.append([])
    #base_estimator = Pipeline([
    #    ('scaler', StandardScaler()),('model', LogisticRegression(penalty='l1',  solver='liblinear'))])
    logistic = LogisticRegression(penalty='l1', solver='liblinear', max_iter=100)
    selector1 = StabilitySelection(
        base_estimator=logistic, lambda_name='C',lambda_grid=np.logspace(-5, -1, 50), threshold=0.75, 
        bootstrap_func='subsample')
    selector1.fit(newData['bootData'],newData['bootLabels'])
    Z_stab1[i].append(selector1.get_support(indices=True))
    numFeatSelected_stab1[i].append(len(selector1.get_support(indices=True)))
    selected_scores1[i].append(selector1.stability_scores_.max(axis=1))



# Second stability selection method