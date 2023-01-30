import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import norm
from typing import Optional
from scipy.optimize import fixed_point
from warnings import warn
from tqdm import tqdm
import matplotlib.pyplot as plt



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
             self.pnl.append(true_value - bid_or_ask)
        elif dir == -1:
            self.pnl.append(bid_or_ask - true_value)
        else:
            self.pnl.append(0)

class uninformed_trader(trader):

    def __init__(self, trade_prob):
        assert trade_prob <= 0.5, "sell and buy prob are equal so therefore must ne smaller than 0.5"
        self.eta = trade_prob
        self.pnl = []

    def trade(self):
        ## does not know true value (no use of true value here)
        return np.random.choice([1, -1, 0], p=[self.eta, self.eta, 1-2*self.eta])

    
class noisy_informed_trader(trader):

    def __init__(self, sigma_noise):
        self.sigma_noise = sigma_noise
        self.pnl = []

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
        self.pnl = []

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



    
################################## Distributions ##################################


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


    def reset(self, centered_at:Optional[float]=None):

        if centered_at is not None:
            self.center = centered_at
        
        self.compute_vec_v(update_history=True)
        self.compute_prior_v(update_history=True)
    

  
        
    
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
                            update_prior:Optional[bool]=True, 
                            update_v_vec:Optional[bool]=True):
                    
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
        #assert np.abs(Pbuy+Psell+Pno-1)<1e-3, "prior proba or order is normalized"

        post = []

        if order_type == 1:

            for i, v in enumerate(self.vec_v):
            #     if v <= Pa:
            #         post.append(self.prior_v[i]*((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-v, scale=sigma_w))))
            #     else:
            #         post.append(self.prior_v[i]*(alpha*(1-norm.cdf(x=Pa-v, scale=sigma_w)) + (1-alpha)*eta))
                post.append(self.prior_v[i]*((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-v, scale=sigma_w))))

            post = np.array(post)/Pbuy

        
        elif order_type == -1:

            for i, v in enumerate(self.vec_v):
            #     if v <= Pb:
            #         post.append(self.prior_v[i]*(alpha*norm.cdf(x=Pb-v, scale=sigma_w) + (1-alpha)*eta))
            #     else:
            #         post.append(self.prior_v[i]*(alpha*norm.cdf(x=Pb-v, scale=sigma_w) + (1-alpha)*eta))
                post.append(self.prior_v[i]*((1-alpha)*eta + alpha*norm.cdf(x=Pb-v, scale=sigma_w)))

            post = np.array(post)/Psell

        else:

            for i, v in enumerate(self.vec_v):
            #     if v < Pb:
            #         post.append(self.prior_v[i]*((1-alpha)*(1-2*eta) + alpha*(1-norm.cdf(x=Pb-v, scale=sigma_w))))
            #     elif (v>=Pb) and (v<Pa):
            #         post.append(self.prior_v[i]*((1-alpha)*(1-2*eta) + alpha*(1-norm.cdf(x=Pb-v, scale=sigma_w) + norm.cdf(x=Pa-v, scale=sigma_w))))
            #     else:
            #         post.append(self.prior_v[i]*((1-alpha)*(1-2*eta) + alpha*norm.cdf(x=Pa-v, scale=sigma_w)))
                post.append(self.prior_v[i]*((1-2*eta)*(1-alpha) + alpha*(norm.cdf(x=Pa-v, scale=sigma_w) - norm.cdf(x=Pb-v, scale=sigma_w))))


            post = np.array(post)/Pno
            



        if update_prior:
            self.prior_v = post
            self.p_history.append(post)
        if update_v_vec:
            self.v_history.append(self.vec_v)

        return post




##   Prior proba of selling

def P_sell(Pb, alpha, eta, sigma_w, vec_v:Optional[list], v_prior:Optional[list], known_value:Optional[float]=None):
    '''
    This is the prior proba of a selling order arriving
    Args:
        - Pb: the bid price to solve (FP eq)
        - V_min: the min value of true value (min of vec_v)
        - V_max: the max value of true value (max of vec_v)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
    '''
    if (known_value is not None):
        assert known_value>0, "known value is negative, cannot compute P_sell"

        if known_value < Pb:
            result = alpha*norm.cdf(x=Pb-known_value, scale=sigma_w) + (1-alpha)*eta
        else:
            result = alpha*(1-norm.cdf(x=known_value-Pb, scale=sigma_w)) + (1-alpha)*eta

    else:

        #convert v into a df for usefulness
        #prior_on_v = pd.DataFrame(data=[vec_v, v_prior]).T.rename(columns={0:"v", 1:"p"})

        ## first sum when Vi < Pb
        #####result = sum([(alpha*norm.cdf(x=Pb-Vi, scale=sigma_w) + (1-alpha)*eta)*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi < Pb])
        
        ## second sum when Vi >= Pb
        #result += sum([(alpha*(1-norm.cdf(x=Vi-Pb, scale=sigma_w)) + (1-alpha)*eta)*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi >= Pb])
        #####result +=  sum([(alpha*norm.cdf(x=Pb-Vi, scale=sigma_w) + (1-alpha)*eta)*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi >= Pb])

        result = (1-alpha)*eta
        for i, v in enumerate(vec_v):
            result += v_prior[i]*norm.cdf(x=Pb-v, scale=sigma_w)*alpha
    
   
    #assert (result>=0-) and (result<=1), "P_sell is not between 0 and 1, problem"

    return result






## fixed point equation for Bid price

def Pb_fp(Pb, alpha, eta, sigma_w, vec_v:Optional[list], v_prior:Optional[list], known_value:Optional[float]=None):
    '''
    This is the fixed point equation for the bid price Pb
    Args:
        - Pb: the bid price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
    '''

    psell = P_sell(Pb=Pb, alpha=alpha, eta=eta, sigma_w=sigma_w, vec_v=vec_v, v_prior=v_prior, known_value=known_value)

    if known_value is not None:
        assert known_value > 0, "known value is negative, cannot compute P_sell"
        if known_value <= Pb:
            result = ((1-alpha)*eta + alpha*norm.cdf(x=Pb-known_value, scale=sigma_w))*known_value
        else:
            result = ((1-alpha)*eta + alpha*(1-norm.cdf(x=known_value-Pb, scale=sigma_w)))*known_value
            
    
    else:

        #convert v into a df for usefulness
        prior_on_v = pd.DataFrame(data=[vec_v, v_prior]).T.rename(columns={0:"v", 1:"p"})

        result = sum([((1-alpha)*eta + alpha*norm.cdf(x=Pb-Vi, scale=sigma_w))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi <= Pb])

        #result += sum([((1-alpha)*eta + alpha*(1-norm.cdf(x=Vi-Pb, scale=sigma_w)))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi > Pb])
        result += sum([((1-alpha)*eta + alpha*norm.cdf(x=Pb-Vi, scale=sigma_w))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi > Pb])

    return result/psell



##   Prior proba of buying


def P_buy(Pa, alpha, eta, sigma_w, vec_v:Optional[list], v_prior:Optional[list], known_value:Optional[float]=None):
    '''
    This is the prior proba of a buying order arriving
    Args:
        - Pa: the ask price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
    '''
    if (known_value is not None):
        assert known_value>0, "known value is negative, cannot compute P_buy"

        if known_value <= Pa:
            result = alpha*(1-norm.cdf(x=Pa-known_value, scale=sigma_w)) + (1-alpha)*eta
        else:
            result = alpha*norm.cdf(x=known_value-Pa, scale=sigma_w) + (1-alpha)*eta

    else:

        #convert v into a df for usefulness
        #prior_on_v = pd.DataFrame(data=[vec_v, v_prior]).T.rename(columns={0:"v", 1:"p"})

        ## first sum when Vi < Pa
        #result = sum([(alpha*(1-norm.cdf(x=Pa-Vi, scale=sigma_w)) + (1-alpha)*eta)*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi <= Pa])
        #####result = sum([(alpha*(1-norm.cdf(x=Pa-Vi, scale=sigma_w)) + (1-alpha)*eta)*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi <= Pa])

        ## second sum when Vi >= Pa
        #result += sum([(alpha*norm.cdf(x=Vi-Pa, scale=sigma_w) + (1-alpha)*eta)*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi > Pa])
        #####result += sum([(alpha*(1-norm.cdf(x=Pa-Vi, scale=sigma_w)) + (1-alpha)*eta)*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi > Pa])
        result = (1-alpha)*eta
        for i, v in enumerate(vec_v):
            result += alpha*(1-norm.cdf(x=Pa-v,scale=sigma_w))*v_prior[i]
        


    if result <0-0.001 or result>1+0.001:
        pass
    #print(f"Pbuy: {result}")
    #assert (result>=0) and (result<=1), "P_sell is not between 0 and 1, problem"

    return result



##   Prior proba of no order


def P_no_order(Pb, Pa, alpha, eta, sigma_w, vec_v, v_prior):
    '''
    Prior proba of no trade happening
    '''

    assert Pa > Pb, "ask is lower than bid"

    prob = (1-alpha)*(1-2*eta) ## part of uninformed traders

    # for i, v in enumerate(vec_v):
    #     if v<= Pb:
    #         prob += v_prior[i]*alpha*(1-norm.cdf(x=Pb-v, scale=sigma_w))
    #     elif (v>Pb) and (v<=Pa):
    #         prob += v_prior[i]*(alpha*(norm.cdf(x=Pa-v, scale=sigma_w) + (1-norm.cdf(x=Pb-v, scale=sigma_w))))
    #     else:
    #         prob += v_prior[i]*alpha*norm.cdf(x=Pa-v, scale=sigma_w)

    #if prob<0 or prob>1:
        #print(f"prob is not corrrect {prob}") 
    #assert (prob>=0) and (prob<=1), "prob is not between 0 and 1"

    for i, v in enumerate(vec_v):

        prob += v_prior[i]*alpha*(norm.cdf(x=Pa-v, scale=sigma_w) - norm.cdf(x=Pb-v, scale=sigma_w))

    #assert (prob>=0) and (prob<=1), "prob is not between 0 and 1"


    return prob




## fixed point equation for ask price

def Pa_fp(Pa, alpha, eta, sigma_w, vec_v:Optional[list], v_prior:Optional[list], known_value:Optional[float]=None):
    '''
    This is the fixed point equation for the ask price Pa
    Args:
        - Pa: the ask price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
    '''

    pbuy = P_buy(Pa=Pa, alpha=alpha, eta=eta, sigma_w=sigma_w, vec_v=vec_v, v_prior=v_prior, known_value=known_value)

    if known_value is not None:
        assert known_value > 0, "known value is negative, cannot compute P_buy"
        if known_value <= Pa:
            result = ((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-known_value, scale=sigma_w)))*known_value
        else:
            result = ((1-alpha)*eta + alpha*norm.cdf(x=known_value-Pa, scale=sigma_w))*known_value
            
    
    else:

        #convert v into a df for usefulness
        prior_on_v = pd.DataFrame(data=[vec_v, v_prior]).T.rename(columns={0:"v", 1:"p"})

        #result = sum([((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-Vi, scale=sigma_w)))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi <= Pa]) 
        result = sum([((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-Vi, scale=sigma_w)))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi <= Pa]) 

        #result += sum([((1-alpha)*eta + alpha*norm.cdf(x=Vi-Pa, scale=sigma_w))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi > Pa])
        result += sum([((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-Vi, scale=sigma_w)))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi > Pa])
                
    return result/pbuy




## expectation of true value

def compute_exp_true_value(Pb, Pa, psell, pbuy, pno, vec_v, v_prior, alpha, eta, sigma_w):

    exp = Pa*psell + Pb*pbuy 

    for i, v in enumerate(vec_v):

        exp += v*v_prior[i]*alpha*(norm.cdf(x=Pa-v, scale=sigma_w) - norm.cdf(x=Pb-v, scale=sigma_w))
    
    return exp





class GMD_simluation():

    '''This class allows to run a full simulation of a GMD model on an fake asset
        - The asset price is a jump process
        - there are 2 types of traders:
            - uninformed traders
            - noisy informed traders (perfectly informed if the sigma_noise is set to very small values)
        - the market maker knows:
            - the initial asset value
            - the probabilstic nature of trading crowd
            - the *occurence* of jumps in the true asset price

       Params:
            - tmax: duration of simulation
            - sigma_price: std of jumps distribution (normal) (responsible with "multiplier" of simulation duration)
            - proba_jump: probability of a jump at any iteration
            - alpha: proportion of (noisy) informed traders
            - eta: probability of a buy/sell order from an uninformed trader
            - sigma_noise: std of noise distribution (normal) of noisy informed trader
            - V0: initial true value
            - multiplier: parameter setting the discretization qtty (responsible with "sigma_price" of simulation duration)
                - default will lead to precision of cents 
                - higher will leads to finer grid (longer simulation)
            - eps_discrete_error: allowed discretization error
        '''


    def __init__(self, 
                    tmax : int, 
                    sigma_price : float, 
                    proba_jump : float, 
                    alpha : float, 
                    eta : float, 
                    sigma_noise : float, 
                    V0 : Optional[float]=100,
                    multiplier : Optional[int]=None, 
                    eps_discrete_error:Optional[float]=1e-4):

        ## atttributes/params of simulation
        self.tmax = tmax
        self.alpha = alpha
        self.sigma_price = sigma_price
        self.proba_jump = proba_jump
        self.eta = eta
        self.sigma_w = sigma_noise
        self.V0 = V0
        self.multiplier = multiplier
        self.eps = eps_discrete_error
        self.run = False

        ## compute price dynamics
        self.get_asset_dynamics()


    def __str__(self):

        res = f"GMD simulation with following params: \n- {self.tmax} iterations\n"
        res += f"- price with jump probability {self.proba_jump} and amplitude std {self.sigma_price} (current path has {self.jumps.count(1)} jumps)\n"
        res += f"- proportion informed traders is {self.alpha}\n"
        res += f"- noise std of informed traders is {self.sigma_w}\n"
        res += f"- probability of random trade by uninformed traders is {self.eta}\n"
        if self.multiplier is None:
            res += f"- V(t=0)={self.V0}, multiplier is set to default"
        else:
            res += f"- V(t=0)={self.V0}, multiplier is {self.multiplier} so prob density len is {int(self.multiplier*2*self.sigma_price*4)}"
        if not self.run:
            res += "\n       Not run yet       "
        else:
            res += "\n       Already run       "
        return res


    

    def get_asset_dynamics(self):
        '''
        This methods simulates one asset price path and saves it for when the 
        a simulation is run
        '''
        
        val = asset_dynamics(p_jump=self.proba_jump, sigma=self.sigma_price, init_price=self.V0)

        val.simulate(tmax=self.tmax)

        self.jumps = val.dynamics["jumps"].to_list()
        self.true_value = val.price(tmax=self.tmax).to_list()




    def run_simulation(self):
        '''
        This methods runs one simulation with the given parameters
        '''

        if self.multiplier is None:
            self.multiplier = int(400/(2*self.sigma_price*4))

        self.v_distrib = Vi_prior(sigma_price=self.sigma_price, centered_at=self.V0, multiplier=self.multiplier)

        self.asks = []
        self.bids = []
        self.exp_value = []

        self.u_trader = uninformed_trader(trade_prob=self.eta)
        self.i_trader = noisy_informed_trader(sigma_noise=self.sigma_w)

        self.traders_order = np.random.choice(["i", "u"], size=self.tmax, p=[self.alpha, 1-self.alpha])

        for t in tqdm([t for t in range(self.tmax)]):

            if self.jumps[t]==1:
                self.v_distrib.reset(centered_at=self.exp_value[-1])
            
            ## MM sets bid and ask price
            self.asks.append(fixed_point(Pa_fp, self.true_value[t], args=(self.alpha, self.eta, self.sigma_w, self.v_distrib.vec_v, self.v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item())
            self.bids.append(fixed_point(Pb_fp, self.true_value[t], args=(self.alpha, self.eta, self.sigma_w, self.v_distrib.vec_v, self.v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item())

            ## priors or buying, selling, no order
            Pbuy = P_buy(Pa=self.asks[-1], alpha=self.alpha, eta=self.eta, sigma_w=self.sigma_w, vec_v=self.v_distrib.vec_v, v_prior=self.v_distrib.prior_v)
            assert Pbuy>0-self.eps and Pbuy<1+self.eps, "Pbuy not between 0 and 1"
            
            Psell = P_sell(Pb=self.bids[-1], alpha=self.alpha, eta = self.eta, sigma_w=self.sigma_w, vec_v=self.v_distrib.vec_v, v_prior=self.v_distrib.prior_v)
            assert Psell>0-self.eps and Psell<1+self.eps, "Psell not between 0 and 1"
            
            Pnoorder = P_no_order(Pb=self.bids[-1], Pa=self.asks[-1], alpha=self.alpha, eta=self.eta, sigma_w=self.sigma_w, vec_v=self.v_distrib.vec_v, v_prior=self.v_distrib.prior_v)
            assert Pnoorder>0-self.eps and Pnoorder<1+self.eps, "P_noorder not between 0 and 1"

            assert Psell+Pbuy+Pnoorder>0-self.eps and Pbuy +Psell+Pnoorder<1+self.eps, "sum of order priors not between 0 and 1"

            ## compute expected value
            self.exp_value.append(compute_exp_true_value(Pb=self.bids[-1], 
                                                    Pa=self.asks[-1],
                                                    psell=Psell, 
                                                    pbuy=Pbuy,
                                                    pno=Pnoorder,
                                                    vec_v=self.v_distrib.vec_v,
                                                    v_prior=self.v_distrib.prior_v,
                                                    alpha=self.alpha,
                                                    eta=self.eta,
                                                    sigma_w=self.sigma_w))

            ## tarders trade
            if self.traders_order[t] == "i":
                trade = self.i_trader.trade(bid=self.bids[-1], ask=self.asks[-1], true_value=self.true_value[t])
                self.u_trader.update_pnl(0, 45, 45)
                ## the noise is added to the true value for the noisy informed trader to trade
            else:
                trade = self.u_trader.trade()
                self.i_trader.update_pnl(0, 45, 45)


            ## update MM proba distribution
            self.v_distrib.compute_posterior(trade, 
                                            Pbuy=Pbuy, 
                                            Psell=Psell, 
                                            Pno=Pnoorder, 
                                            Pa=self.asks[-1], 
                                            Pb=self.bids[-1],
                                            alpha=self.alpha, 
                                            eta=self.eta, 
                                            sigma_w=self.sigma_w,
                                            update_prior=True)

            assert np.abs(sum(self.v_distrib.prior_v)-1) < self.eps, "posterior prob is not normalized"

        self.run = True
        print("Simulation finsihed")



    def show_result(self):
        '''
        This methods shows plots of:
            - the true value and the expected value over time
            - bid/ask prices over time
            - spread over time
            - MM true value distribution at 3 different iterations
        '''


        fig, ax = plt.subplots(2,2, figsize=(10,8))

        bids=np.array(self.bids)
        asks=np.array(self.asks)
        

        ax[0, 1].plot(self.bids, label="bid price")
        ax[0, 1].plot(self.asks, label="ask price")
        ax[0,1].legend()
        ax[0,1].set_xlabel("time t")

        ax[0,0].plot(self.true_value, label="True value")
        ax[0,0].set_ylabel("asset value")
        ax[0,0].plot(self.exp_value, label="Exp. value")
        ax[0,0].legend()
        ax[0,0].set_ylim(ax[0,1].get_ylim())
        ax[0,0].set_xlabel("time t")

        ax[1,0].plot((asks-bids)/(0.5*(asks+bids)), label="spread")
        ax[1,0].set_xlabel("time t")
        ax[0,1].set_ylabel("bid/ask")
        ax[1,0].set_ylabel("spread")

        #snapshots_i = (int(self.tmax/3), int(2*self.tmax/3), int(3*self.tmax/3-2))
        snapshots_i = [3,6,9]
        for snap in snapshots_i:
            ax[1,1].plot(self.v_distrib.v_history[snap], self.v_distrib.p_history[snap], label=f"iter: {snap}")
        ax[1,1].legend()
        ax[1,1].set_xlabel("true values")
        ax[1,1].set_ylabel("count (normalized)")

        #ax[2,1].plot(self.i_trader.pnl)
        #ax[1,2].plot(self.i_trader.pnl)
        fig.tight_layout()

        plt.show()




    
    


