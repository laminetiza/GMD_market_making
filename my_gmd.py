import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy.stats import norm
from typing import Optional
from scipy.optimize import fixed_point
from warnings import warn
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats



'''
Here are some classes useful for the purpose of Glosten-Milgrom-Das model (in order)
It includes: 
    - different classes of traders
    - class representing the asset dynamic
    - class representing probability densities over possibles values (priors, posterios) with a method ot update it
    - functions (without class) computing prior probabilites of trades (Pbuy, Psell, Pnoorder), 2 functions to solve FP equations implied by 
    the condition that BID = E[V | Sell order] and ASK = E[V | Buy order] in GMD model (Das's paper). there is also a function to comptue the price (expected value)
    - class of Glosten Milgrom Das model simulation with a fake asset with methods to run it and show results
    - class of GMD model simulation of profit per unit time made by the MM when its spread is extended by a fixed amount
    - class of Glosten Milgrom Das model simulation with real asset price (method to load it and infer useful params based on it)
'''

################################## TRADERS ##################################

class trader(ABC):
    '''
    Abstract class representing a trader
    '''

    @abstractmethod
    def trade(self, bid : float, ask : float, true_value : float):
        pass
    
    def update_pnl(self, direction : int, bid_or_ask : float, true_value : float):
        if direction == 1:
             self.pnl.append(true_value - bid_or_ask)
        elif dir == -1:
            self.pnl.append(bid_or_ask - true_value)
        else:
            self.pnl.append(0)


class uninformed_trader(trader):
    '''
    Class representing an uninformed trader of the GMD model
    Args:
        - trade_prob: probability to buy/sell (max is 0.5 since it is both the buy 
        and sell prob and 1-2*eta is the proba that no order is passed)
    '''

    def __init__(self, trade_prob : float):
        assert trade_prob <= 0.5, "sell and buy prob are equal so therefore must ne smaller than 0.5"
        self.eta = trade_prob
        self.pnl = []

    def trade(self) -> int:
        '''
        the trader randomly trades
        '''
        ## does not know true value (no use of true value here)
        return np.random.choice([1, -1, 0], p=[self.eta, self.eta, 1-2*self.eta])

    

class noisy_informed_trader(trader):
    '''
    Class representing a (noisy) informed trader in the GMD model
    Args:
        - sigma_noise: std of gaussian distribution representing the noise 
    '''

    def __init__(self, sigma_noise : float):
        self.sigma_noise = sigma_noise
        self.pnl = []


    def trade(self, bid : float, ask : float, true_value : float) -> int:
        '''
        The noisy informed trader trades
        '''
        ## the informed trader knows the true value

        noisy_value = true_value + np.random.normal(0, self.sigma_noise)

        if noisy_value > ask:
            return 1
        elif (bid < noisy_value) and (noisy_value < ask):
            return 0
        else:
            return -1
    

class perfectly_informed_trader(trader):
    '''
    Class representing a perfectly informed trader in the GMD model
    '''

    def __init__(self):
        self.pnl = []


    def trade(self, bid : float, ask : float, true_value : float) -> int:
        '''
        The trader trades
        '''

        if true_value > ask:
            return 1
        elif (bid < true_value) and (true_value < ask):
            return 0
        else:
            return -1



################################## Asset dynamics ##################################


class asset_dynamics():
    '''
    This class implements the fake asset price described in Das Paper 
    link: https://cs.gmu.edu/~sanmay/papers/das-qf-rev3.pdf
    
    Args:
        - p_jump: probability of a jump occuring at a each iteration
        - sigma: std of the gaussian distribution representing the amplitude (and direction) of the jump if it occurs
        - init_price: initial price 
    '''

    def __init__(self, 
                    p_jump : float, 
                    sigma : float, 
                    init_price: float):

        self.std = sigma
        self.p0 = init_price
        self.p_jump = p_jump
        self.dynamics = None

    
    def simulate(self, tmax : int):
        '''
        This methods simulates a price path
        Args:
            - tmax: duration of simulation
        '''

        data = np.zeros(tmax) 
        data[0] = self.p0 ## initial value
        data = pd.DataFrame(data)
        
        # initial value is not a jump. It is then random according to p_jump
        jumps = [0] + list(np.random.choice([1, 0], size=tmax-1, p=[self.p_jump, 1-self.p_jump]))
    
        data["jumps"] = jumps
        data["amp"] = np.random.normal(0, self.std, tmax)
        data["reals"] = data.apply(lambda se: se[0] + se["jumps"]*se["amp"], axis=1) # add amplitude only where jumps occurs
        data["price"] = data.reals.cumsum(0)

        ## object now has a "current" dynamics
        self.dynamics = data


    def price(self, tmax  : int) -> pd.Series:
        '''
        This methods returns the simulated price (re-simulates it with tmax if not the same as current 
        simulation tmax (if any current simulation))
        Args: 
            - tmax: duration of simulation
        Returns: 
            - fake asset price timeseries
        '''

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
    It is initialized as a gaussian (discrete vector of probabilities) and
    can be updated at all times with new trade information coming in
    '''

    def __init__(self, 
                    sigma_price : float, 
                    centered_at : float, 
                    multiplier : int, 
                    nb_sigma_range : Optional[int] = 4):
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
        self.nb_sigma_range = nb_sigma_range
        self.vec_v = None
        self.prior_v = None
        self.v_history = []
        self.p_history = []
        self.compute_vec_v()
        self.compute_prior_v()


    def reset(self, centered_at:Optional[float]=None):
        '''
        This methods resets the prior distribution 
        Args: 
            - centered_at: if provided, the new distribution will be recented at this value 
        '''

        if centered_at is not None:
            self.center = centered_at
        
        self.compute_vec_v(update_history=True) ## those 2 methods store the new vectors 
        self.compute_prior_v(update_history=True)
    

  
        
    
    def compute_vec_v(self, update_history:Optional[bool]=True):
        '''
        This method creates the vector of possible values of v
        If the center changed: need to update it beforehand
        
        Args: 
            - update_history: if True (default), the vector is added to a list with previous 
            vectors to keep track of them 
        '''

        vec_v = []
        for i in range(int(2*self.nb_sigma_range*self.sigma*self.multiplier+1)):
            vec_v.append(self.center-self.nb_sigma_range*self.sigma+i/self.multiplier)
        
        self.vec_v = vec_v

        if update_history:
            self.v_history.append(vec_v)


    def compute_prior_v(self, update_history:Optional[bool]=True):
        '''
        This method creates the vector of probas for v

        Args:
            - update_history: if True (default), the vector is added to a list with previous 
            vectors to keep track of them 
        '''

        prior_v = []
        for i in range(int(2*self.nb_sigma_range*self.sigma*self.multiplier+1)):
            prior_v.append(norm.cdf(x=-self.nb_sigma_range*self.sigma+(i+1)/self.multiplier, scale=self.sigma) - norm.cdf(x=-self.nb_sigma_range*self.sigma+i/self.multiplier, scale=self.sigma))
        
        self.prior_v = prior_v
        if update_history:
            self.p_history.append(prior_v)


    
    def compute_posterior(self, 
                            order_type : int, 
                            Pbuy : float, 
                            Psell : float, 
                            Pno : float, 
                            Pa : float, 
                            Pb : float, 
                            alpha : float, 
                            eta : float, 
                            sigma_w : float, 
                            update_prior : Optional[bool]=True, 
                            update_v_vec : Optional[bool]=True) -> list:
                    
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
        '''

        assert Pb < Pa, "ask price is below bid price"

        post = []

        if order_type == 1:

            for i, v in enumerate(self.vec_v):
                post.append(self.prior_v[i]*((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-v, scale=sigma_w))))

            post = np.array(post)/Pbuy

        
        elif order_type == -1:

            for i, v in enumerate(self.vec_v):
                post.append(self.prior_v[i]*((1-alpha)*eta + alpha*norm.cdf(x=Pb-v, scale=sigma_w)))

            post = np.array(post)/Psell

        else:

            for i, v in enumerate(self.vec_v):
                post.append(self.prior_v[i]*((1-2*eta)*(1-alpha) + alpha*(norm.cdf(x=Pa-v, scale=sigma_w) - norm.cdf(x=Pb-v, scale=sigma_w))))

            post = np.array(post)/Pno
            
        if update_prior:
            self.prior_v = post
            self.p_history.append(post)
        if update_v_vec:
            self.v_history.append(self.vec_v)

        return post




##   Prior proba of selling

def P_sell(Pb : float, 
            alpha : float, 
            eta : float, 
            sigma_w : float, 
            vec_v:Optional[list], 
            v_prior:Optional[list], 
            known_value:Optional[float]=None) -> float:
    '''
    This is the prior proba of a selling order arriving
    Args:
        - Pb: the bid price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi

    Returns: 
        - prior proba of the MM receiving a sell order
    '''
    if (known_value is not None):
        assert known_value>0, "known value is negative, cannot compute P_sell"

        if known_value < Pb:
            result = alpha*norm.cdf(x=Pb-known_value, scale=sigma_w) + (1-alpha)*eta
        else:
            result = alpha*(1-norm.cdf(x=known_value-Pb, scale=sigma_w)) + (1-alpha)*eta

    else:

        result = (1-alpha)*eta
        for i, v in enumerate(vec_v):
            result += v_prior[i]*norm.cdf(x=Pb-v, scale=sigma_w)*alpha
    
    return result






## fixed point equation for Bid price

def Pb_fp(Pb : float, 
            alpha : float, 
            eta : float, 
            sigma_w : float, 
            vec_v:Optional[list], 
            v_prior:Optional[list], 
            known_value:Optional[float]=None) -> float:
    '''
    This is the fixed point equation for the bid price Pb
    Args:
        - Pb: the bid price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
    Ooutput: 
        - bid price sollution to fixed point equation 
    '''

    ## compute prior proba of sell order
    psell = P_sell(Pb=Pb, alpha=alpha, eta=eta, sigma_w=sigma_w, vec_v=vec_v, v_prior=v_prior, known_value=known_value)

    if known_value is not None:
        assert known_value > 0, "known value is negative, cannot compute P_sell"
        if known_value <= Pb:
            result = ((1-alpha)*eta + alpha*norm.cdf(x=Pb-known_value, scale=sigma_w))*known_value
        else:
            result = ((1-alpha)*eta + alpha*(1-norm.cdf(x=known_value-Pb, scale=sigma_w)))*known_value
            
    else:

        prior_on_v = pd.DataFrame(data=[vec_v, v_prior]).T.rename(columns={0:"v", 1:"p"})

        result = sum([((1-alpha)*eta + alpha*norm.cdf(x=Pb-Vi, scale=sigma_w))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi <= Pb])

        result += sum([((1-alpha)*eta + alpha*norm.cdf(x=Pb-Vi, scale=sigma_w))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi > Pb])

    return result/psell



##   Prior proba of buying

def P_buy(Pa : float, 
            alpha : float, 
            eta : float, 
            sigma_w : float, 
            vec_v: list, 
            v_prior: list, 
            known_value:Optional[float]=None) -> float:
    '''
    This is the prior proba of a buying order arriving
    Args:
        - Pa: the ask price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
    Output:
        - prior proba of buy order
    '''

    if (known_value is not None):
        assert known_value>0, "known value is negative, cannot compute P_buy"

        if known_value <= Pa:
            result = alpha*(1-norm.cdf(x=Pa-known_value, scale=sigma_w)) + (1-alpha)*eta
        else:
            result = alpha*norm.cdf(x=known_value-Pa, scale=sigma_w) + (1-alpha)*eta

    else:

        result = (1-alpha)*eta
        for i, v in enumerate(vec_v):
            result += alpha*(1-norm.cdf(x=Pa-v,scale=sigma_w))*v_prior[i]

    return result



##   Prior proba of no order


def P_no_order(Pb : float,
                Pa : float, 
                alpha : float, 
                eta : float, 
                sigma_w : float, 
                vec_v : list, 
                v_prior : list) -> float:
    '''
    This is the prior proba of a no order arriving
    Args:
        - Pb: the bid price 
        - Pa: the ask price 
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
    Output:
        - prior proba of no order 
    '''

    assert Pa > Pb, "ask is lower than bid"

    prob = (1-alpha)*(1-2*eta) ## part of uninformed traders

    for i, v in enumerate(vec_v):

        prob += v_prior[i]*alpha*(norm.cdf(x=Pa-v, scale=sigma_w) - norm.cdf(x=Pb-v, scale=sigma_w))

    return prob




## fixed point equation for ask price

def Pa_fp(Pa : float, 
            alpha : float,
            eta : float, 
            sigma_w : float, 
            vec_v: list, 
            v_prior: list, 
            known_value:Optional[float]=None) -> float:
    '''
    This is the fixed point equation for the ask price Pa
    Args:
        - Pa: the ask price to solve (FP eq)
        - alpha: prob of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi

    Output:
        - the ask price solution of the FP equation
    '''
    
    
    # prior proba of buying order arriving
    pbuy = P_buy(Pa=Pa, alpha=alpha, eta=eta, sigma_w=sigma_w, vec_v=vec_v, v_prior=v_prior, known_value=known_value)

    if known_value is not None:
        assert known_value > 0, "known value is negative, cannot compute P_buy"
        if known_value <= Pa:
            result = ((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-known_value, scale=sigma_w)))*known_value
        else:
            result = ((1-alpha)*eta + alpha*norm.cdf(x=known_value-Pa, scale=sigma_w))*known_value
            
    
    else:

        prior_on_v = pd.DataFrame(data=[vec_v, v_prior]).T.rename(columns={0:"v", 1:"p"})

        result = sum([((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-Vi, scale=sigma_w)))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi <= Pa]) 

        result += sum([((1-alpha)*eta + alpha*(1-norm.cdf(x=Pa-Vi, scale=sigma_w)))*Vi*(prior_on_v[prior_on_v["v"]==Vi]["p"].item()) for Vi in vec_v if Vi > Pa])
                
    return result/pbuy




## expectation of true value

def compute_exp_true_value(Pb : float, 
                            Pa : float, 
                            psell : float, 
                            pbuy : float, 
                            vec_v : list, 
                            v_prior : list, 
                            alpha : float,
                            eta : float,
                            sigma_w : float) -> float:
    '''
    This methods compute the expected value of the asset 
    Args:
        - Pb: bid price
        - Pa: ask price
        - psell: prior proba of sell order
        - pbuy: prior proba of buy oder
        - vec_v: vector of value Vi
        - v_prior: prior probability of V=Vi
        - alpha: proportion of informed traders
        - eta: proportion of buy/sell orders from uninformed traders
        - sigma_w: std of noise of noisy informed traders 
    
    Output:
        - expected value
    '''

    exp = Pa*psell + Pb*pbuy ## expected value conditionned on buying order, selling order

    for i, v in enumerate(vec_v):
        ## expected value conditionned on no order arriving (was not already computed)
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
            - extend_spread: amount (in cents) to add to ask and remove from bid to steer free from zero profit condition for MM
            - gamma: inventory control coefficient
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
                    eps_discrete_error:Optional[float]=1e-4, 
                    extend_spread: Optional[float] = 0,
                    gamma:Optional[float]=0):

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
        self.extend_spread = extend_spread
        self.gamma = gamma
        
        ## compute price dynamics
        self.get_asset_dynamics()


    def __str__(self) -> str:
        '''
        method printing summary of simulation params
        '''

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
        This methods runs one simulation with the given parameters (if already run, this 
        method will overwrite previous results)
        '''

        if self.multiplier is None:
            self.multiplier = int(400/(2*self.sigma_price*4))

        ## initialize prior
        self.v_distrib = Vi_prior(sigma_price=self.sigma_price, centered_at=self.V0, multiplier=self.multiplier)

        self.asks = []
        self.bids = []
        self.exp_value = []
        self.pnl = []
        self.inventory = [0]

        ## initialize trading crowd and traders order
        self.u_trader = uninformed_trader(trade_prob=self.eta)
        self.i_trader = noisy_informed_trader(sigma_noise=self.sigma_w)
        self.traders_order = np.random.choice(["i", "u"], size=self.tmax, p=[self.alpha, 1-self.alpha])

        
        for t in tqdm([t for t in range(self.tmax)]):
            
            ## if jump, reset the prior distribution on V
            if self.jumps[t]==1:
                self.v_distrib.reset(centered_at=self.exp_value[-1])
            
            ## MM sets bid and ask price
            curr_ask = fixed_point(Pa_fp, self.true_value[t], args=(self.alpha, self.eta, self.sigma_w, self.v_distrib.vec_v, self.v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item()
            curr_ask += -self.gamma*self.inventory[-1]
            curr_ask += self.extend_spread ## extend spread
            self.asks.append(curr_ask)

            curr_bid = fixed_point(Pb_fp, self.true_value[t], args=(self.alpha, self.eta, self.sigma_w, self.v_distrib.vec_v, self.v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item()
            curr_bid += -self.gamma*self.inventory[-1]
            curr_bid += -self.extend_spread ## extend spread
            self.bids.append(curr_bid)


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
                                                    vec_v=self.v_distrib.vec_v,
                                                    v_prior=self.v_distrib.prior_v,
                                                    alpha=self.alpha,
                                                    eta=self.eta,
                                                    sigma_w=self.sigma_w))

            ## tarders trade
            if self.traders_order[t] == "i":
                trade = self.i_trader.trade(bid=self.bids[-1], ask=self.asks[-1], true_value=self.true_value[t])
                self.u_trader.update_pnl(0, 45, 45) ## not useful (abitrary)
            else:
                trade = self.u_trader.trade()
                self.i_trader.update_pnl(0, 45, 45) ## not useful (abitrary)

            ## Update MM pnl
            self.update_pnl_and_inventory(trader_direction=trade, bid=self.bids[-1], ask=self.asks[-1], true_value=self.true_value[t])


            ## update MM proba distribution with trade info
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

        self.run = True # simulation finised
        print("Simulation finsihed")



    def show_result(self, 
                    figsize:Optional[tuple]=(10,9),
                    dpi:Optional[int]=100, 
                    same_y_axis_above:Optional[bool]=True):
        '''
        This methods shows plots of:
            - the true value and the expected value over time
            - bid/ask prices over time
            - spread over time
            - MM true value distribution at 3 different iterations
        '''


        fig, ax = plt.subplots(2,2, figsize=(figsize[0],figsize[1]), dpi=dpi)

        bids=np.array(self.bids)
        asks=np.array(self.asks)
        

        ax[0, 1].plot(self.bids, label="bid price", alpha=0.8)
        ax[0, 1].plot(self.asks, label="ask price", alpha=0.8)
        ax[0,1].legend()
        ax[0,1].set_xlabel("time t")

        ax[0,0].plot(self.true_value, label="True value", alpha=0.8)
        ax[0,0].set_ylabel("asset value")
        ax[0,0].plot(self.exp_value, label="Exp. value", alpha=0.8)
        ax[0,0].legend()
        if same_y_axis_above:
            ax[0,0].set_ylim(ax[0,1].get_ylim())
        ax[0,0].set_xlabel("time t")

        # ax[1,0].plot((asks-bids)/(0.5*(asks+bids)), label="spread")
        ax[1,0].plot((asks-bids), label="asbolute spread", alpha=0.8)

        ax[1,0].set_xlabel("time t")
        ax[0,1].set_ylabel("bid/ask")
        ax[1,0].set_ylabel("absolute spread")

        #snapshots_i = (int(self.tmax/3), int(2*self.tmax/3), int(3*self.tmax/3-2))
        snapshots_i = [3,6,9]
        for snap in snapshots_i:
            ax[1,1].plot(self.v_distrib.v_history[snap], self.v_distrib.p_history[snap], label=f"iter: {snap}", alpha=0.8)
        ax[1,1].legend()
        ax[1,1].set_xlabel("true values")
        ax[1,1].set_ylabel("count (normalized)")

        fig.tight_layout()

        plt.show()


    

    def show_true_value(self):
        '''
        Shows the current true value path of the simulation object
        '''
        plt.plot(self.true_value)
        plt.xlabel("time t")
        plt.ylabel("value")
        plt.show()



    def update_pnl_and_inventory(self, 
                                    trader_direction : int, 
                                    bid : float, 
                                    ask : float, 
                                    true_value : float):
        '''
        This methods updates the inventory of the MM and its pnl
        Args:
            - trader_direction: int representing buy (1), sell(-1) or no order (0)
            - bid: bid price
            - ask : ask price
            - true_value : true value of the fake asset
        '''

        if trader_direction == 1:
            ## The MM is short
            self.pnl.append(ask-true_value)
            self.inventory.append(self.inventory[-1] - 1)

        elif trader_direction == -1:
            ## the MM is long
            self.pnl.append(true_value-bid)
            self.inventory.append(self.inventory[-1] + 1)

        else:
            self.pnl.append(0)
            self.inventory.append(self.inventory[-1])





class Spread_after_jump_analysis():
    '''
    This class is only used a few times to compute the decay of MM spread after a jump occurs 
    '''


    def __init__(self, 
                    number_of_simulations : int,
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
        self.nb_simuls = number_of_simulations 




    def get_asset_dynamics(self):
        '''
        This methods simulates one asset price path and saves it for when the 
        a simulation is run
        '''
        
        val = asset_dynamics(p_jump=self.proba_jump, sigma=self.sigma_price, init_price=self.V0)

        val.simulate(tmax=self.tmax)

        self.jumps = val.dynamics["jumps"].to_list()
        self.true_value = val.price(tmax=self.tmax).to_list()




    def run_simulation(self) -> list:
        '''
        Runs the simulation. Very similar execution as the one of the class above but tracks the spread after a jump 
        and store list of spreads after a jump in jumps_spreads list
        '''

        jumps_spreads = [] 

        if self.multiplier is None:
            multiplier = int(400/(2*self.sigma_price*4))

        for simul in tqdm([i for i in range(self.nb_simuls)]):
            
            # create a price path
            val = asset_dynamics(p_jump=self.proba_jump, sigma=self.sigma_price, init_price=self.V0)
            val.simulate(tmax=self.tmax)
            jumps = val.dynamics["jumps"].to_list()
            true_value = val.price(tmax=self.tmax).to_list()

            # if no jump, stop this iteration, else
            if jumps.count(1) > 0:

                v_distrib = Vi_prior(sigma_price=self.sigma_price, centered_at=self.V0, multiplier=self.multiplier)
                asks = []
                bids = []
                exp_value = []
                u_trader = uninformed_trader(trade_prob=self.eta)
                i_trader = noisy_informed_trader(sigma_noise=self.sigma_w)
            
                traders_order = np.random.choice(["i", "u"], size=self.tmax, p=[self.alpha, 1-self.alpha])
                curr_level_spread = []

                for t in range(self.tmax):

                    if jumps[t]==1:
                        v_distrib.reset(centered_at=exp_value[-1])
                        ## record spread after this jump and reset current spread to empty list
                        jumps_spreads.append(curr_level_spread)
                        curr_level_spread = []
                        
                        
                    ## MM sets bid and ask price
                    asks.append(fixed_point(Pa_fp, true_value[t], args=(self.alpha, self.eta, self.sigma_w, v_distrib.vec_v, v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item())
                    bids.append(fixed_point(Pb_fp, true_value[t], args=(self.alpha, self.eta, self.sigma_w, v_distrib.vec_v, v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item())
                    curr_level_spread.append(asks[-1]-bids[-1])

                    ## priors or buying, selling, no order
                    Pbuy = P_buy(Pa=asks[-1], alpha=self.alpha, eta=self.eta, sigma_w=self.sigma_w, vec_v=v_distrib.vec_v, v_prior=v_distrib.prior_v)
                    assert Pbuy>0-self.eps and Pbuy<1+self.eps, "Pbuy not between 0 and 1"
                    
                    Psell = P_sell(Pb=bids[-1], alpha=self.alpha, eta = self.eta, sigma_w=self.sigma_w, vec_v=v_distrib.vec_v, v_prior=v_distrib.prior_v)
                    assert Psell>0-self.eps and Psell<1+self.eps, "Psell not between 0 and 1"
                    
                    Pnoorder = P_no_order(Pb=bids[-1], Pa=asks[-1], alpha=self.alpha, eta=self.eta, sigma_w=self.sigma_w, vec_v=v_distrib.vec_v, v_prior=v_distrib.prior_v)
                    assert Pnoorder>0-self.eps and Pnoorder<1+self.eps, "P_noorder not between 0 and 1"

                    assert Psell+Pbuy+Pnoorder>0-self.eps and Pbuy +Psell+Pnoorder<1+self.eps, "sum of order priors not between 0 and 1"

                    ## compute expected value
                    exp_value.append(compute_exp_true_value(Pb=bids[-1], 
                                                            Pa=asks[-1],
                                                            psell=Psell, 
                                                            pbuy=Pbuy,
                                                            vec_v=v_distrib.vec_v,
                                                            v_prior=v_distrib.prior_v,
                                                            alpha=self.alpha,
                                                            eta=self.eta,
                                                            sigma_w=self.sigma_w))

                    ## tarders trade
                    if traders_order[t] == "i":
                        trade = i_trader.trade(bid=bids[-1], ask=asks[-1], true_value=true_value[t])
                        u_trader.update_pnl(0, 45, 45)
                        ## the noise is added to the true value for the noisy informed trader to trade
                    else:
                        trade = u_trader.trade()
                        i_trader.update_pnl(0, 45, 45)


                    ## update MM proba distribution
                    v_distrib.compute_posterior(trade, 
                                                    Pbuy=Pbuy, 
                                                    Psell=Psell, 
                                                    Pno=Pnoorder, 
                                                    Pa=asks[-1], 
                                                    Pb=bids[-1],
                                                    alpha=self.alpha, 
                                                    eta=self.eta, 
                                                    sigma_w=self.sigma_w,
                                                    update_prior=True)

                    assert np.abs(sum(v_distrib.prior_v)-1) < self.eps, "posterior prob is not normalized"


        return jumps_spreads





class GMD_simluation_ext_data():

    '''This class allows to run a full simulation of a GMD model on an true asset
        - The asset price is from a true crypto or another asset
        - there are 2 types of traders:
            - uninformed traders
            - noisy informed traders (perfectly informed if the sigma_noise is set to very small values)
        - the market maker knows:
            - the initial asset value
            - the probabilstic nature of trading crowd
            - the *occurence* of jumps in the true asset price

       Params:
            - tmax: duration of simulation
            - alpha: proportion of (noisy) informed traders
            - eta: probability of a buy/sell order from an uninformed trader
            - sigma_noise: std of noise distribution (normal) of noisy informed trader
            - datapath: path containing parquet file of asset price 
            - multiplier: parameter setting the discretization qtty (responsible with "sigma_price" of simulation duration)
                - default will lead to precision of cents 
                - higher will leads to finer grid (longer simulation)
            - nb_sigma_range: how much std's should one side of prior distribution cover 
                - if too small: the range of values covered by discrete prior distribution is too small
            - boost_sigma_price: multiplier of the std to cover a to broader range of values with a probability less close to 0 
            - threshold_jump: thresold (in stds) above which a return is a "Jump" (the MM knows the jumps)
            - eps_discrete_error: allowed discretization error
            - extend_spread: amount (in cents) to add to ask and remove from bid to steer free from zero profit condition for MM
            - gamma: inventory control coefficient
        '''


    def __init__(self, 
                    tmax : int,
                    alpha : float, 
                    eta : float, 
                    sigma_noise : float, 
                    data_path : str, 
                    multiplier : Optional[int]=None, 
                    nb_sigma_range : Optional[int] = 5, 
                    boost_sigma_price: Optional[float] = 1,
                    threshold_jump: Optional[float] = 1,
                    eps_discrete_error:Optional[float]=1e-4, 
                    extend_spread: Optional[float] = 0,
                    gamma:Optional[float]=0):

        ## atttributes/params of simulation
        self.alpha = alpha
        self.eta = eta
        self.sigma_w = sigma_noise
        self.multiplier = multiplier
        self.nb_sigma_range = nb_sigma_range
        self.eps = eps_discrete_error
        self.boost_sigma_price = boost_sigma_price
        self.threshold_jump = threshold_jump
        self.run = False
        self.extend_spread = extend_spread
        self.gamma = gamma
        self.read_data(datapath=data_path, tmax=tmax)

    
    def read_data(self, datapath, tmax):
         
        data = pd.read_parquet(datapath).tail(tmax)
        data.price = data.price*100/data.price.head(1).item()

        self.sigma_price = data.price.diff(1).std()
        self.sigma_price = self.sigma_price * self.boost_sigma_price
        if self.boost_sigma_price != 1:
            print(f"Boosting std by {self.boost_sigma_price}")

        print(f"sigma price= {self.sigma_price}")
        self.V0 = data.price.head(1).item()
        print(f"V0= {self.V0}")
        self.tmax = len(data)
        self.true_value = data.price.to_list()

        ## infer jumps (introduces lookahead bias)
        self.jumps = data.price.diff(1).apply(lambda re: 1 if np.abs(re)>self.threshold_jump*self.sigma_price else 0).to_list()
        self.jumps = [0]+self.jumps ## first price is not a "jump"


    

    def run_simulation(self):
        '''
        This methods runs one simulation with the given parameters
        '''

        if self.multiplier is None:
            self.multiplier = int(400/(2*self.sigma_price*4))

        self.v_distrib = Vi_prior(sigma_price=self.sigma_price, centered_at=self.V0, multiplier=self.multiplier, nb_sigma_range=self.nb_sigma_range)
        print(f"vec_v from {self.v_distrib.vec_v[0]} to {self.v_distrib.vec_v[-1]} with len {len(self.v_distrib.vec_v)}")

        self.asks = []
        self.bids = []
        self.exp_value = []
        self.pnl = []
        self.inventory = [0]

        self.u_trader = uninformed_trader(trade_prob=self.eta)
        self.i_trader = noisy_informed_trader(sigma_noise=self.sigma_w)

        self.traders_order = np.random.choice(["i", "u"], size=self.tmax, p=[self.alpha, 1-self.alpha])

        for t in tqdm([t for t in range(self.tmax)]):

            if self.jumps[t]==1:
                self.v_distrib.reset(centered_at=self.exp_value[-1])
            
            ## MM sets bid and ask price
            curr_ask = fixed_point(Pa_fp, self.true_value[t], args=(self.alpha, self.eta, self.sigma_w, self.v_distrib.vec_v, self.v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item()
            #curr_ask += -self.gamma*self.inventory[-1]
            curr_ask += self.extend_spread ## extend spread
            self.asks.append(curr_ask)

            curr_bid = fixed_point(Pb_fp, self.true_value[t], args=(self.alpha, self.eta, self.sigma_w, self.v_distrib.vec_v, self.v_distrib.prior_v), xtol=1e-2, maxiter=500, method='del2').item()
            #curr_bid += -self.gamma*self.inventory[-1]
            curr_bid += -self.extend_spread ## extend spread
            self.bids.append(curr_bid)


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

            ## Update MM pnl
            self.update_pnl_and_inventory(trader_direction=trade, bid=self.bids[-1], ask=self.asks[-1], true_value=self.true_value[t])


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





    def show_result(self, 
                    figsize:Optional[tuple]=(10,9),
                    dpi:Optional[int]=100, 
                    same_y_axis_above:Optional[bool]=True):
        '''
        This methods shows plots of:
            - the true value and the expected value over time
            - bid/ask prices over time
            - spread over time
            - MM true value distribution at 3 different iterations
        '''


        fig, ax = plt.subplots(2,2, figsize=(figsize[0],figsize[1]), dpi=dpi)


        bids=np.array(self.bids)
        asks=np.array(self.asks)
        

        ax[0, 1].plot(self.bids, label="bid price", alpha=0.8)
        ax[0, 1].plot(self.asks, label="ask price", alpha=0.8)
        ax[0,1].legend()
        ax[0,1].set_xlabel("time t")

        ax[0,0].plot(self.true_value, label="True value", alpha=0.8)
        ax[0,0].set_ylabel("asset value")
        ax[0,0].plot(self.exp_value, label="Exp. value", alpha=0.8)
        for i in range(self.tmax):
            if self.jumps[i] ==1:
                ax[0,0].axvline(i, alpha=0.1, color="k")
        ax[0,0].legend()
        if same_y_axis_above:
            ax[0,0].set_ylim(ax[0,1].get_ylim())
        ax[0,0].set_xlabel("time t")

        # ax[1,0].plot((asks-bids)/(0.5*(asks+bids)), label="spread")
        ax[1,0].plot((asks-bids), label="asbolute spread", alpha=0.8)

        ax[1,0].set_xlabel("time t")
        ax[0,1].set_ylabel("bid/ask")
        ax[1,0].set_ylabel("absolute spread")

        #snapshots_i = (int(self.tmax/3), int(2*self.tmax/3), int(3*self.tmax/3-2))
        snapshots_i = [3,6,9]
        for snap in snapshots_i:
            ax[1,1].plot(self.v_distrib.v_history[snap], self.v_distrib.p_history[snap], label=f"iter: {snap}", alpha=0.8)
        ax[1,1].legend()
        ax[1,1].set_xlabel("true values")
        ax[1,1].set_ylabel("count (normalized)")

        fig.tight_layout()

        plt.show()


    

    def show_true_value(self, with_jumps:Optional[bool]=False):
        '''
        plots the current true value path
        Args:
            - with jumps: if True, will shade vertically on timestamps where a jump *occured*

        '''

        plt.plot(self.true_value)
        if with_jumps:
            for i in range(self.tmax):
                if self.jumps[i] ==1:
                    plt.axvline(i, alpha=0.1, color="k")
        plt.xlabel("time t")
        plt.ylabel("value")
        plt.show()




    def update_pnl_and_inventory(self, 
                                    trader_direction : int, 
                                    bid : float, 
                                    ask : float, 
                                    true_value : float):
        '''
        This methods updates the inventory of the MM and its pnl
        Args:
            - trader_direction: int representing buy (1), sell(-1) or no order (0)
            - bid: bid price
            - ask : ask price
            - true_value : true value of the fake asset
        '''
                

        if trader_direction == 1:
            ## The MM is short
            self.pnl.append(ask-true_value)
            self.inventory.append(self.inventory[-1] - 1)

        elif trader_direction == -1:
            ## the MM is long
            self.pnl.append(true_value-bid)
            self.inventory.append(self.inventory[-1] + 1)

        else:
            self.pnl.append(0)
            self.inventory.append(self.inventory[-1])