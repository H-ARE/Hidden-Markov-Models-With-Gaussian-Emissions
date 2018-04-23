import numpy as np
import pprint
from tools2 import *
from prondict import prondict
import matplotlib.pyplot as plt
phoneHMMs = np.load('lab2_models.npz')['phoneHMMs'].item()
from sklearn.mixture import log_multivariate_normal_density
pp = pprint.PrettyPrinter(indent=4)
pp = pprint.PrettyPrinter(width=41, compact=True)
import warnings
warnings.filterwarnings("ignore")
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
        # print(mean)
        # print("")
        means[i:i+3,0:13]=mean
        covars[i:i+3,0:13]=cov
        i+=3
    transmat[-1,-1]=1.0
    startprobs=np.zeros(tsize)
    startprobs[0]=1.0
    # print(covars)
    # print("HEJ HEK")
    # print(means)
    combinedHMM={'covars':covars,'name':namelist[0],'transmat':transmat,'startprob':startprobs,'means':means}
    return combinedHMM


def main():

    a=concatHMMs(phoneHMMs,namelist=prondict['4'])
    data = np.load('lab2_data.npz')['data'][10]
    #loglik=example['obsloglik']
    #print()
    fakelog=log_multivariate_normal_density(data['lmfcc'],a['means'],a['covars'])
    # plt.pcolormesh(example['lmfcc'])
    # plt.show()

    #4.1
    # plt.pcolormesh(fakelog.transpose())
    # plt.colorbar()
    # plt.show()

    #4.2
    log_alpha = forward(fakelog, np.log(a['startprob']), np.log(a['transmat']))

    #4.3
    # x = viterbi(fakelog, np.log(a['startprob']), np.log(a['transmat']))
    # print(x)

    #4.4
    log_beta = backward(fakelog, np.log(a['startprob']), np.log(a['transmat']))

    #5.1
    # print(log_alpha)
    # print("------------------------------")
    # print(log_beta)
    log_gamma = statePosteriors(log_alpha, log_beta)

    #5.2
    mu, covar = updateMeanAndVar(data['lmfcc'],log_gamma)
    #new_fake_log=log_multivariate_normal_density(data['lmfcc'],mu,covar)
    new_fake_log = fakelog;

    for x in range(1,10):

        log_alpha = forward(new_fake_log, np.log(a['startprob']), np.log(a['transmat']))
        log_beta = backward(new_fake_log, np.log(a['startprob']), np.log(a['transmat']))
        log_gamma = statePosteriors(log_alpha, log_beta)
        mu, covar = updateMeanAndVar(data['lmfcc'],log_gamma)
        new_fake_log=log_multivariate_normal_density(data['lmfcc'],mu,covar)
        print(new_fake_log[0,0])

    print(new_fake_log)
    print(fakelog)



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
    alpha = np.zeros(log_emlik.shape)

    alpha[0,:] = log_startprob[0:-1] + log_emlik[0,:]
    #print(len(log_emlik))
    sum_row = 0;
    log_transmat = log_transmat[0:-1];

    # for state in range(1,9):
    #     alpha[1,:] = logsumexp(alpha[0,:] + log_transmat[:,state]) + log_emlik[1,state]

    for frame in range(1,len(log_emlik)):
        
        for state in range(0,len(log_emlik[0])):

            alpha[frame,state] = logsumexp(alpha[frame-1,:] + log_transmat[:,state]) + log_emlik[frame,state]
        

    # pp.pprint(alpha)
    return alpha
def backward(log_emlik, log_startprob, log_transmat):
    """Backward (beta) probabilities in log domain.

    Args:
        log_emlik: NxM array of emission log likelihoods, N frames, M states
        log_startprob: log probability to start in state i
        log_transmat: transition log probability from state i to j

    Output:
        backward_prob: NxM array of backward log probabilities for each of the M states in the model
    """

    beta = np.zeros(log_emlik.shape)
    n = len(log_emlik)-2
    log_transmat = log_transmat[0:-1,0:-1];


    # beta[69,0] = logsumexp(log_transmat[0,:] + log_emlik[70,:] + beta[70,:])

    while n >= 0:
        
        for j in range(0,len(log_emlik[0])):

            beta[n,j] = logsumexp(log_transmat[j,:] + log_emlik[n+1,:] + beta[n+1,:])

        n = n -1
    # pp.pprint(beta)

    return beta


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
    vit = np.zeros(log_emlik.shape)


    vit[0] = log_startprob[0:-1] + log_emlik[0,:]
    log_transmat = log_transmat[0:-1];


    for n in range(1,len(log_emlik)):
        
        for j in range(0,len(log_emlik[0])):

            vit[n,j] = max(vit[n-1,:] + log_transmat[:,j]) + log_emlik[n,j]
        

    # pp.pprint(vit)
    # pp.pprint(vit.argmax(1))
    return (vit[-1,vit.argmax(1)[-1]], vit.argmax(1))


def statePosteriors(log_alpha, log_beta):
    """State posterior (gamma) probabilities in log domain.

    Args:
        log_alpha: NxM array of log forward (alpha) probabilities
        log_beta: NxM array of log backward (beta) probabilities
    where N is the number of frames, and M the number of states

    Output:
        log_gamma: NxM array of gamma probabilities for each of the M states in the model
    """

    gamma = np.zeros(log_alpha.shape)

    gamma = log_alpha + log_beta 

    gamma = gamma - logsumexp(log_alpha[-1,:])
    #gamma = np.exp(log_alpha) + np.exp(log_beta)
    #print(gamma)

    #pp.pprint(gamma)

    return gamma
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

    gamma_shape = log_gamma.shape
    x_shape = X.shape
    gamma = np.exp(log_gamma);

    mu = np.zeros([gamma_shape[1],x_shape[1]])
    covar = np.zeros([gamma_shape[1],x_shape[1]])

    for j in range(gamma_shape[1]):  #state
        temp = 0;
        for n in range(x_shape[0]):  #dim
            temp = temp + gamma[n,j]*X[n,:]

        mu[j,:] = temp/np.sum(gamma[:,j])

    # print(mu)
    # print(mu.shape)

    for j in range(gamma_shape[1]):  #state
        temp = 0;
        for n in range(x_shape[0]):  #dim
            temp = temp + gamma[n,j]*(X[n,:] - mu[j,:])*(X[n,:] - mu[j,:]).transpose()

        covar[j,:] = temp/np.sum(gamma[:,j])
    # print(covar)
    # print(covar.shape)

    return mu, covar

#namelist=['sil', 'ow', 'sil']

if __name__ == '__main__':
    main()


