import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in log")

class ExponMLE():
    ''' INPUTS:
        start, stop (float): range of parameters to check for MLE
        data (array): array of potentially exponential random variables
        
        NOTE: this class assumes the exponential function:
        y = 1/beta * exp(-x/beta), with mean = beta
        scipy.stats.expon(scale=beta)
    '''
    def __init__(self, start, stop, data):
        self.start = start
        self.stop = stop
        self.data = data
        
        self.min = np.min(data)
        self.max = np.max(data)

    def expon_log_likeli(self, c):
        '''determine likelihood of a parameter c given the data'''
        dist = stats.expon(scale=c)
        likelihood = dist.pdf(self.data)
        log_likeli = np.log(likelihood)
        return np.sum(log_likeli)

    def expon_mle(self):
        '''determine parameter with maximum likelihood over a range of parameters'''
        candidates = np.linspace(self.start, self.stop, 100)
        res = []
        for c in candidates:
            res.append(self.expon_log_likeli(c))
        mx_idx = np.argmax(res)
        mle = candidates[mx_idx]
        return mle, res

    def plot_mle(self, ax):
        '''plot results of search for MLE'''
        ax.plot(np.linspace(self.start, self.stop, 100), self.expon_mle()[1],
                color='k', linestyle=':', linewidth=3, label='log likelihoods')
        ax.axhline(y=np.max(self.expon_mle()[1]), color='m', linewidth=2, 
                   label=f'ML: {np.max(self.expon_mle()[1]):2.2f}')
        ax.set_title(f'MLE: {self.expon_mle()[0]:2.2f}',fontsize = 25)
        ax.set_yticks([])
        ax.set_xlabel('units of time', fontsize = 20)
        ax.legend(fontsize = 20, loc='center')
        
    def plot_dist(self, ax):
        '''plot histogram of actual data overlayed with exponential pdf from MLE'''
        x = np.linspace(self.min, self.max, 1000)
        y = stats.expon(scale = self.expon_mle()[0]).pdf(x)
        ax.hist(data, density=True, color='yellowgreen', label='data')
        ax.plot(x, y, color='m', 
                label = f'exponential pdf (beta={self.expon_mle()[0]:2.2f})')
        ax.set_yticks([])
        ax.set_xlabel('units of time', fontsize = 20)
        ax.set_title('MLE Distribution Fit to Data', fontsize = 25)
        ax.legend(fontsize = 20, loc='center')
        
if __name__=="__main__":
    
    # dummy, reproducible, well-behaved expon data
    np.random.seed(1111)
    data = stats.expon(scale=12).rvs(10**3)

    # instantiate (very narrow) exponential class
    exp_mle = ExponMLE(0.1, 100, data)
    exp_mle.expon_mle()

    # plot MLE for beta
    fig, ax = plt.subplots(figsize=(10,5))
    exp_mle.plot_mle(ax)

    # plot histogram of real data and pdf of data from MLE estimation
    fig, ax = plt.subplots(figsize=(10,5))
    exp_mle.plot_dist(ax)

    plt.show()