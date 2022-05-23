import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn import metrics
import data_tools
import stability_multiscales as sm 


# Read plink data
data = data_tools.data_reader('bed', 'bim', 'fam')

# Read the LD_groups 
LD_groups = data_tools.read_LDgroups('LD_groups.csv')

# Read the genes
genes = data_tools.read_genes('genes.csv')
# Obtain the genes IDs
genes_names = list(set(genes.Gene))



M = 10
num_ratios = 10
num_alphas = 10
alpha = 0.05 
alphas = np.linspace(0.05,0.95,num_alphas) ## the first hyper-parameter of the elastic net
ratios = np.linspace(0.01,0.8,num_ratios)  ## the second hyper-parameter of the elastic net
Z_net = np.zeros((num_ratios,num_alphas,M,data.shape[1]),dtype=np.int8)
errors_net = np.zeros((M,num_ratios,num_alphas))
stabilities_net = np.zeros((num_ratios,num_alphas))
errStab_net = np.zeros((num_ratios,num_alphas))
numFeatSelected_net = np.zeros((M,num_ratios,num_alphas),dtype=np.int32)

for i in range(M):
    newData = sm.getBootstrapSample(data,labels) ## we get bootstrap samples
    for k in range(num_ratios):
        for l in range(num_alphas):
            net = ElasticNet(alpha = alphas[l], l1_ratio = ratios[k], max_iter=500) ## we use elastic net
            net.fit(newData['bootData'],newData['bootLabels'])
            Z_net[k,l,i,net.coef_!= 0] = 1
            numFeatSelected_net[i,k,l] = np.sum(Z_net[k,l,i,],axis=0)
            predLabels_net = np.zeros(len(newData['OOBData']))
            temp = net.predict(newData['OOBData'])
            for j in range(len(newData['OOBData'])):
                if temp[j] >= 0: predLabels_net[j] = 1
                else:  predLabels_net[j] = -1
            errors_net[i,k,l] = 1-metrics.accuracy_score(newData['OOBLabels'], predLabels_net)
meanError_net = np.mean(errors_net,0)

# Compute the stability at the SNP level
    