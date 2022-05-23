import numpy as np
import math


def pearson(s1,s2):

    """ Computes the Pearson's correlation coeffient between two arrays s1 and a list s2 """
    
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

    """ 
    Compute the stability of the selection index.
    Args: 
        A : list of lists of selected SNPs for 10 subsamples

        func : the similarity index used to compute the stability index (such as pearson correlation)

    Returns: 
        stability : value of stability index
        
    """
    """ Computes the average pairwise similarities between the rows of A """
    M = np.size(A,0)
    stability = 0
    for i in range(M):
        for j in range(M):
            if i != j:
                stability = stability + func(A[i], A[j])
    
    return(stability/(np.dot(M,(M-1))))


def stability_LD(LD_label, LD_group):
	
    
    stability_list = []
    for i in range(len(LD_label)):
        L = [0] * LD_group.LD.max()
        for j in LD_label[i]:
            L[j-1] = 1
        stability_list.append(L)
        
    return stability_list
 
def snp2LD_index(Z, snps_list, LD_group):
    
    selected_snps = []
    LD_label = []
    res_LD_pearson = []; 
    for k in range(num_lambdas):
        selected_snps.append([])
        LD_label.append([])
        for l in range(num_bootstrap):
            selected_snps[k].append([])
            LD_label[k].append([])
            for m in range(len(snps_list)):
                if Z[k,l,m] == 1:
                    selected_snps[k][l].append(snps[m])
                    LD_label[k][l].append(LD_group.LD.values[m])
                else:
                    continue                        
        res_LD_pearson.append(stab_index(stability_LD(LD_label[k], LD_group)))
    LDgroups_flat = [[item for sublist in x for item in sublist] for x in LD_label]
    LDgroups_flat_nodup = [list(set(i)) for i in LDgroups_flat]
    numLDgroupSelected = [len(i) for i in LDgroups_flat_nodup]
    return res_LD_pearson, selected_snps, numLDgroupSelected


def snp2gene_index(selected_snps, genes):
    genes_label = []
    for k in range(num_lambdas):
        genes_label.append([])
        for l in range(num_bootstrap):
            genes_label[k].append([])
            for m in range(len(selected_snps[k][l])):
                if selected_snps[k][l][m] in list(genes.SNP):
                    ind = list(genes.SNP).index(selected_snps[k][l][m])
                    genes_label[k][l].append(genes.Gene.values[ind])
                else:
                    continue
    selected_genes = [[list(set(i)) for i in x] for x in genes_label]
    genes_flat = [[item for sublist in x for item in sublist] for x in selected_genes]
    genes_flat_nodup = [list(set(i)) for i in genes_flat]
    numGeneSelected = [len(i) for i in genes_flat_nodup]
    return genes_label, selected_genes


def genes_stab(genes_IDs, selected_genes):
    stability_list = np.array([[[0]*len(genes_IDs)]*num_bootstrap]*num_lambdas)
    res_genes_pearson = []; 
    for k in range(num_lambdas):
        for l in range(num_bootstrap):
            for m in range(len(selected_genes[k][l])):
                if selected_genes[k][l][m] in genes_IDs:                
                    ind = genes_IDs.index(selected_genes[k][l][m])
                    stability_list[k][l][ind] = 1
                else:
                    continue
        res_genes_pearson.append(stab_index(stability_list[k]))
    return res_genes_pearson