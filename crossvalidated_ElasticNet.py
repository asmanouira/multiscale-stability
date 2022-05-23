import numpy as np
import math
import pandas as pd
from sklearn import metrics
from pandas_plink import read_plink1_bin
from sklearn.linear_model import LogisticRegressionCV



def pearson(s1,s2):
    d = len(s1)
    ki = np.sum(s1)
    kj = np.sum(s2)
    expR = np.dot(ki,kj)/d 
    pi = ki/d
    pj = kj/d
    upsiloni = np.sqrt(np.dot(pi,(1-pi)))
    upsilonj = np.sqrt(np.dot(pj,(1-pj)))
    sum_list = [a + b for a, b in zip(s1, s2)]
    r = sum_list.count(2)
    similarity = (r-expR)/(d*upsiloni*upsilonj)
    if (ki==d and kj==d) or (ki==0 and kj==0):
        similarity = 1
    elif math.isnan(similarity):
        similarity = 0
    return(similarity)

def stab_index(A,func = pearson):

    M = np.size(A,0)
    stability = 0
    for i in range(M):
        for j in range(M):
            if i != j:
                stability = stability + func(A[i], A[j])
    stability = stability/(np.dot(M,(M-1)))
    return(stability)
def stability_LD(LD_label, LD_group):

    stability_list = []
    for i in range(len(LD_label)):
        L = [0] * LD_group.LD.max()
        for j in LD_label[i]:
            L[j-1] = 1
        stability_list.append(L)
        
    return stability_list
def snp2LD_index(Z, snps_list, LD_group):
    
    selected_snps = []; LD_label = []; res_LD_pearson = []; 
    for l in range(M):
        selected_snps.append([])
        LD_label.append([])
        for m in range(len(snps_list)):
            if Z[l,m] == 1:
                selected_snps[l].append(snps[m])
                LD_label[l].append(LD_group.LD.values[m])
            else:
                continue                        
    res_LD_pearson.append(stab_index(stability_LD(LD_label, LD_group)))
    LDgroups_flat = [item for item in LD_label] 
    LDgroups_flat_nodup = [list(set(i)) for i in LDgroups_flat]
    numLDgroupSelected = [len(i) for i in LDgroups_flat_nodup]

    return res_LD_pearson, selected_snps, numLDgroupSelected
def snp2gene_index(selected_snps, genes):
    genes_label = []
    for l in range(M):
        genes_label.append([])
        for m in range(len(selected_snps[l])):
            if selected_snps[l][m] in list(genes.SNP):
                ind = list(genes.SNP).index(selected_snps[l][m])
                genes_label[l].append(genes.Gene.values[ind])
            else:
                continue
    selected_genes = [list(set(i)) for i in genes_label] 
    return genes_label, selected_genes
def genes_stab(genes_IDs, selected_genes):
    stability_list = np.array([[0]*len(genes_IDs)]*M); res_genes_pearson = []; 
    for l in range(M):
        for m in range(len(selected_genes[l])):
            if selected_genes[l][m] in genes_IDs:                
                ind = genes_IDs.index(selected_genes[l][m])
                stability_list[l][ind] = 1
            else:
                continue
    res_genes_pearson.append(stab_index(stability_list))
    return res_genes_pearson
def getBootstrapSample(data,labels):

    m,d=data.shape
    if m!= len(labels):
        raise ValueError('The data and labels should have a same number of rows.')
    ind=np.random.choice(range(m), size=m, replace=True)
    OOBind=np.setdiff1d(range(m),ind, assume_unique=True)
    bootData=data[ind,]
    bootLabels=labels[ind]
    OOBData=data[OOBind,]
    OOBLabels=labels[OOBind]
    return {'bootData':bootData,'bootLabels':bootLabels,'OOBData':OOBData,'OOBLabels':OOBLabels}

## Read data 
plink = read_plink1_bin('myra.bed', 'myra.bim', 'myra.fam')
data = plink.values
data[np.isnan(data)] = 0
labels = list(map(int, plink.trait.values))
snps = list(plink.variant.snp.values)
labels = np.array(labels)
LD_groups = pd.read_csv('LD.csv', sep=',')
genes = pd.read_csv('genes.csv', sep=',')

## Initialize number of bootstraps M and the output arrays
M=10
Z_net=np.zeros((M,data.shape[1]),dtype=np.int8)
errors_net=np.zeros(M)
stabilities_net=np.zeros(M)
errStab_net=np.zeros(M)
best_alpha=np.zeros(M)
best_ratio_l1=np.zeros(M)
numFeatSelected_net=np.zeros((M),dtype=np.int32)

for i in range(M):

    newData=getBootstrapSample(data,labels) ## we get bootstrap samples
    net=LogisticRegressionCV(penalty = 'elasticnet', solver = 'saga', l1_ratios = np.linspace(0.01,0.8,10) cv=5, n_jobs = 10) ## we use elastic net
    net.fit(newData['bootData'],newData['bootLabels'])
    best_alpha[i]=net.C_
    best_ratio_l1[i]=net.l1_ratio_
    Z_net[i,:] = sum(net.coef_!= 0)
    numFeatSelected_net[i]=np.sum(Z_net[i,],axis=0)
    predLabels_net=np.zeros(len(newData['OOBData']))
    predLabels_net=net.predict(newData['OOBData'])
    errors_net[i]=1-metrics.accuracy_score(newData['OOBLabels'], predLabels_net)


SNP_stab = stab_index(Z_net)
LD_stab, sel_snps, numselectedLDgroups = snp2LD_index(Z_net, snps, LD_groups)
genes_label, sel_genes = snp2gene_index(sel_snps, genes)
genes_names = list(set(genes.Gene))
gene_stab = genes_stab(genes_names,sel_genes)

np.savetxt('numFeatSelected_ra_net.txt', numFeatSelected_net)
np.savetxt('errors_ra_net.txt', errors_net)
np.savetxt('best_alpha_net_ra.txt', best_alpha)
np.savetxt('best_ratio_l1_net_ra.txt', best_ratio_l1)
np.savetxt('LD_stab_net_ra.txt', LD_stab)
np.savetxt('SNP_stab_net_ra.txt, SNP_stab')
np.savetxt('gene_stab_net_ra.txt', gene_stab)
np.savetxt('selectedGenes_net_ra.txt', sel_genes)
np.savetxt('numLDgroups_net_ra.txt', numselectedLDgroups)