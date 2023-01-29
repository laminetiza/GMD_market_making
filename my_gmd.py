import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import norm
from typing import Optional


'''
Here are some classes useful for the purpose of Glosten-Milgrom-Das model
It includes: 
    - different classes of traders
    - class representing the asset dynamic
    - class representing probability densities over possibles values (priors, posterios) to be updated
'''

################################## TRADERS ##################################

class trader(ABC):

    @abstractmethod
    def trade(self, bid, ask, true_value):
        pass
    
    def update_pnl(self, direction, bid_or_ask, true_value):
        if direction == 1:
             self.pnl += true_value - bid_or_ask
        elif dir == -1:
            self.pnl += bid_or_ask - true_value
        else:
            pass

class uninformed_trader(trader):

    def __init__(self, trade_prob):
        assert trade_prob <= 0.5, "sell and buy prob are equal so therefore must ne smaller than 0.5"
        self.eta = trade_prob
        self.pnl = 0

    def trade(self):
        ## does not know true value (no use of true value here)
        return np.random.choice([1, -1, 0], p=[self.eta, self.eta, 1-2*self.eta])

    
class noisy_informed_trader(trader):

    def __init__(self, sigma_noise):
        self.sigma_noise = sigma_noise
        self.pnl = 0

    def trade(self, bid, ask, true_value):
        ## the informed trader knows the true value

        noisy_value = true_value + np.random.normal(0, self.sigma_noise)

        if noisy_value > ask:
            return 1
        elif (bid < noisy_value) and (noisy_value < ask):
            return 0
        else:
            return -1
    

class perfectly_informed_trader(trader):

    def __init__(self):
        self.pnl = 0

    def trade(self, bid, ask, true_value):

        if true_value > ask:
            return 1
        elif (bid < true_value) and (true_value < ask):
            return 0
        else:
            return -1

    


################################## Asset dynamics ##################################


class asset_dynamics():

    def __init__(self, p_jump, sigma, init_price):

        self.std = sigma
        self.p0 = init_price
        self.p_jump = p_jump
        self.dynamics = None

    
    def simulate(self, tmax):

        data = np.zeros(tmax)
        data[0] = self.p0
        data = pd.DataFrame(data)
        jumps = [0] + list(np.random.choice([1, 0], size=tmax-1, p=[self.p_jump, 1-self.p_jump]))
        data["jumps"] = jumps
        data["amp"] = np.random.normal(0, self.std, tmax)
        data["reals"] = data.apply(lambda se: se[0] + se["jumps"]*se["amp"], axis=1)
        data["price"] = data.reals.cumsum(0)

        self.dynamics = data

        #return data

    def price(self, tmax):

        if self.dynamics is None:
            self.simulate(tmax=tmax)["price"]
            return self.dynamics["price"]
        elif tmax == len(self.dynamics):
            return self.dynamics["price"]
        else: 
            self.simulate(tmax=tmax)
            return self.dynamics["price"]



    
################################## Distribution ##################################


class Vi_prior():
    '''
    This class represents the prior probability on the true value V
    that the market maker keeps updated at all times
    It is initialized as a gaussian (discrete vector of values) and
    can be updated at all times with new trade information coming in
    '''

    def __init__(self, sigma_price, centered_at, multiplier):
        '''
        Args:
            - sigma_price: the std of the asset returns distribution 
            - centered_at: value around which the discrete vector of proba
            is centered (can be initialized to true value at t=0)
            - multiplier: setting the length of vector values according to 
            len=2*4*sigma_price*multiplier
        '''
        self.sigma = sigma_price
        self.center = centered_at
        self.multiplier = multiplier
        self.vec_v = None
        self.prior_v = None
        self.v_history = []
        self.p_history = []
        self.compute_vec_v()
        self.compute_prior_v()
  
        
    
    def compute_vec_v(self, update_history:Optional[bool]=True):
        '''
        This method creates the vector of possible values of v
        If the center changed: need to update it beforehand
        '''
        vec_v = []
        for i in range(int(2*4*self.sigma*self.multiplier+1)):
            vec_v.append(self.center-4*self.sigma+i/self.multiplier)
        
        self.vec_v = vec_v
        if update_history:
            self.v_history.append(vec_v)
        

    def compute_prior_v(self, update_history:Optional[bool]=True):
        '''
        This method creates the vector of probas for v
        '''
        prior_v = []
        for i in range(int(2*4*self.sigma*self.multiplier+1)):
            prior_v.append(norm.cdf(x=-4*self.sigma+(i+1)/self.multiplier, scale=self.sigma) - norm.cdf(x=-4*self.sigma+i/self.multiplier, scale=self.sigma))
        
        self.prior_v = prior_v
        if update_history:
            self.p_history.append(prior_v)


    
    def compute_posterior(self, 
                            order_type:int, 
                            Pbuy:float, 
                            Psell:float, 
                            Pno:float, 
                            Pa:float, 
                            Pb:float, 
                            alpha:float, 
                            eta:float, 
                            sigma_w:float, 
                            update_prior:Optional[bool]=True):
                    
        '''
        This methods computes the posterior or P(V=Vi) when trade info is received
        It will update the Market Maker belief of the true value
        Args:
            - order_type: 1, -1 or 0 respectively for buy, sell, or no order 
            (no order also contains information)
            - Pbuy: prior proba of receiving a buy order
            - Psell: prior proba of receiving a sell order
            - Pno: prior proba of receiving no order
            - Pa: ask price of the transaction
            - Pb: bid price of the transaction
            - alpha: proportion of informed trader (perfectly informed or noisy informed)
            - eta: proba of buy/sell order for uninformed trader (again, the market maker is 
            aware of the probabilstic structure of trading agents)
            - sigma_w: std of noise distribution (gaussian) of noisy informed traders
            - update_prior: if True (default), will update the prior (as well as prior history) 
        Output:
            will return the posterior distribution 
        
        Note:
            not all arguments of this method are used, due to different calculation 
            depending in order type 
            also, the code is not bullet-proof but this is not the goal, I thus put the 
            min number of "asserts" to ensure minimum consistency
        '''

        assert Pb < Pa, "ask price is below bid price"

        post = []

        if order_type == 1:

            for i, v in enumerate(self.vec_v):
                if v <= Pa:
                    post.append(self.prior_v[i]*((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-v, scale=sigma_w))))
                else:
                    post.append(self.prior_v[i]*(alpha*(1-norm.cdf(x=Pa-v, scale=sigma_w)) + (1-alpha)*eta))

            post = np.array(post)/Pbuy

        
        elif order_type == -1:

            for i, v in enumerate(self.vec_v):
                if v <= Pb:
                    post.append(self.prior_v[i]*(alpha*norm.cdf(x=Pb-v, scale=sigma_w) + (1-alpha)*eta))
                else:
                    post.append(self.prior_v[i]*(alpha*norm.cdf(x=Pb-v, scale=sigma_w) + (1-alpha)*eta))
            
            post = np.array(post)/Psell

        else:

            for i, v in enumerate(self.vec_v):
                if v < Pb:
                    post.append(self.prior_v[i]*((1-alpha)*(1-2*eta) + alpha*(1-norm.cdf(x=Pb-v, scale=sigma_w))))
                elif (v>=Pb) and (v<Pa):
                    post.append(self.prior_v[i]*((1-alpha)*(1-2*eta) + alpha*(1-norm.cdf(x=Pb-v, scale=sigma_w) + norm.cdf(x=Pa-v, scale=sigma_w))))
                else:
                    post.append(self.prior_v[i]*((1-alpha)*(1-2*eta) + alpha*norm.cdf(x=Pa-v, scale=sigma_w)))

            post = np.array(post)/Pno
            
        
        #assert np.abs(sum(post)-1)<1e-4, "proba distr is not normalized"
        print(sum(post))

        if update_prior:
            self.prior_v = post
            self.p_history.append(post)

        return post

