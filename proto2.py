import numpy as np
from tools2 import *
from prondict import prondict
import matplotlib.pyplot as plt
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
from sklearn.mixture import log_multivariate_normal_density
def concatHMMs(hmmmodels, namelist):
    """ Concatenates HMM models in a left to right manner

    Args:
       hmmmodels: list of dictionaries with the following keys:
           name: phonetic or word symbol corresponding to the model
           startprob: M+1 array with priori probability of state
           transmat: (M+1)x(M+1) transition matrix
           means: MxD array of mean vectors
           covars: MxD array of variances
       namelist: list of model names that we want to concatenate

    D is the dimension of the feature vectors
    M is the number of states in each HMM model (could be different for each)

    Output
       combinedhmm: dictionary with the same keys as the input but
                    combined models

    Example:
       wordHMMs['o'] = concatHMMs(phoneHMMs, ['sil', 'ow', 'sil'])
    """
    #output har samma features som modelinputen
    #använd phoneHMMs för att hämta markovmodellerna.
    #namelist är modellerna vi vill kombinera till en modell som vi sedan returnerar
    #modellist = {}
    #for digit in prondict.keys():
    #    modellist[digit] = ['sil'] + prondict[digit] + ['sil']
    names=['sil']+namelist+['sil']
    tsize=3*len(names)+1
    transmat=np.zeros([tsize,tsize])
    i=0
    means=np.zeros([len(names)*3,13])
    covars=np.zeros([len(names)*3,13])
    for digit in names:
        tmat=phoneHMMs[digit]['transmat']
        transmat[i:i+4,i:i+4]=tmat
        mean=phoneHMMs[digit]['means']
        cov=phoneHMMs[digit]['covars']
        #print(cov)
        #print("HEJ HEJ")
        #print(mean)
        means[i:i+3,0:13]=mean
        covars[i:i+3,0:13]=cov
        i+=3
    transmat[-1,-1]=1
    startprobs=np.zeros(tsize)
    startprobs[0]=1
    print(covars)
    print("HEJ HEK")
    print(means)
    combinedHMM={'covars':covars,'name':namelist[0],'transmat':transmat,'startprob':startprobs,'means':means}
    return combinedHMM
a=concatHMMs(phoneHMMs,namelist=prondict['o'])
example = np.load('lab2_example.npz')['example'].item()
loglik=example['obsloglik']
fakelog=log_multivariate_normal_density(example['lmfcc'],a['means'],a['covars'])








def gmmloglik(log_emlik, weights):
    """Log Likelihood for a GMM model based on Multivariate Normal Distribution.

    Args:
        log_emlik: array like, shape (N, K).
            contains the log likelihoods for each of N observations and
            each of K distributions
        weights:   weight vector for the K components in the mixture

    Output:
        gmmloglik: scalar, log likelihood of data given the GMM model.
    """

def forward(log_emlik, log_startprob, log_transmat):
    """Forward (alpha) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: log transition probability from state i to j

    Output:
        forward_prob: NxM array of forward log probabilities for each of the M states in the model
    """

def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

def viterbi(log_emlik, log_startprob, log_transmat):
    """Viterbi path.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        viterbi_loglik: log likelihood of the best path
        viterbi_path: best path
    """

def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

def updateMeanAndVar(X, log_gamma):
    """ Update Gaussian parameters with diagonal covariance

    Args:
         X: NxD array of feature vectors
         log_gamma: NxM state posterior probabilities in log domain
    were N is the lenght of the observation sequence, D is the
    dimensionality of the feature vectors and M is the number of
    states in the model

    Outputs:
         means: MxD mean vectors for each state
         covars: MxD covariance (variance) vectors for each state
    """


namelist=['sil', 'ow', 'sil']


