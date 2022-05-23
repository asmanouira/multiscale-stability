import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import data_tools
import stability_multiscales as sm 

## Read data

# Read plink data
data = data_tools.data_reader('bed', 'bim', 'fam')

# Read the LD_groups 
LD_groups = data_tools.read_LDgroups('LD_groups.csv')

# Read the genes
genes = data_tools.read_genes('genes.csv')
# Obtain the genes IDs
genes_names = list(set(genes.Gene))


## First, we pick the hyper-parameters
num_bootstrap = 10
alpha = 0.05 # the level of significance for confidence intervals and hypothesis tests
num_lambdas = 10
lambdas = np.linspace(-2,-1,num_lambdas)
lambdas = np.power(10,lambdas) ## this gives us lambda values between 10e-2 and 10e-1 (lambda is the regularising parameter)
Z = np.zeros((num_lambdas, num_bootstrap, data.shape[1]),dtype=np.int8)
errors = np.zeros((num_bootstrap, num_lambdas))
stabilities = np.zeros(num_lambdas)
errStab = np.zeros(num_lambdas)
numFeatSelected = np.zeros((num_bootstrap,num_lambdas),dtype=np.int32)
## for each repeat
for i in range(num_bootstrap):
    newData = data_tools.getBootstrapSample(data,labels) ## we get a bootstrap sample
    for k in range(num_lambdas):
        logistic = LogisticRegression(penalty = 'l1', solver = 'liblinear', C = lambdas[k], max_iter = 100) ## We use logistic LASSO
        logistic.fit(newData['bootData'],newData['bootLabels'])  ## we fit the coefficients 
        Z[k,i,:] = sum(logistic.coef_!= 0)
        numFeatSelected[i,k]=sum(sum(logistic.coef_!= 0))
        predLabels = logistic.predict(newData['OOBData'])
        errors[i,k] = 1-metrics.accuracy_score(newData['OOBLabels'], predLabels)
meanError = np.mean(errors,0)
errCI = norm.ppf(1-alpha/2)*(np.std(errors,0))/math.sqrt(num_bootstrap)

# Compute the stability at the SNP level
stab_snp = np.zeros(num_lambdas)
for k in range(num_lambdas):
    stab_snp[k] = sm.stab_index(Z[k,:,:])
# Compute the stability at the LD_groups level 
# Obtain the selected snps and the number of LD_groups selected
stab_LD_groups, sel_snps, numLD = sm.snp2LD_index(Z, snps, LD_groups)
# Obtain the selected genes
lab_g, selected_genes = snp2gene_index(sel_snps,genes)
# Compute the stability at the gene level
stab_gene = genes_stab(genes_names, selected_genes)