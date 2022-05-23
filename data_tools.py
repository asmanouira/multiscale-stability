import numpy as np
import pandas as pd
from pandas_plink import read_plink1_bin


def data_reader(bed, bim, fam):
    plink = read_plink1_bin(bed, bim, fam)
    data = plink.values
    # Fill missing values 
    data[np.isnan(data)] = 0
    # Extract the phenotype
    labels = list(map(int, plink.trait.values))
    labels = np.array(labels)
    # Extract list of snps for further analysis
    snps = list(plink.variant.snp.values)
    return data, labels, snps
    

def getBootstrapSample(data,labels):
    '''
    This function takes as input the genotype data and labels and returns 
    a bootstrap sample of the data, as well as its out-of-bag (OOB) data
    
    INPUTS:
    - data is a 2-dimensional numpy.ndarray where rows are examples and columns are features
    - labels is a 1-dimansional numpy.ndarray giving the label of each example in data
    
    OUPUT:
    - a dictionnary where:
          - key 'bootData' gives a 2-dimensional numpy.ndarray which is a bootstrap sample of data
          - key 'bootLabels' is a 1-dimansional numpy.ndarray giving the label of each example in bootData
          - key 'OOBData' gives a 2-dimensional numpy.ndarray the OOB examples
          - key 'OOBLabels' is a 1-dimansional numpy.ndarray giving the label of each example in OOBData
    '''
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



def read_LDgroups(file, sep):

    """ 
    Read the clustred labels of the LD_groups containing strongly correlated SNPs.

    Args: 

        file: file name that contains two main columns: 
            SNP: str: SNPs IDS such as rdID. Make sure the column name is SNP
            LD: int : label of the LD_group. Make sure the colum name is LD
        sep: seperator operator between element 

    Return:

        LD_groups: pandas dataframe


    """
    LD_groups = pd.read_csv(file , sep)

    return LD_groups

def read_genes(file , sep):

    """
    Read the genes file matching the SNPs to their mapped genes

    Args: 

            file: file name that contains two main columns: 
                SNP: str: SNPs IDS such as rdID. Make sure the column name is SNP
                Gene:  str: genes IDs. Make sure the colum name is Gene
            sep: seperator operator between element 
    Return:

        genes : pandas dataframe
    """

    genes = pd.read_csv(file , sep = sep)

    return genes