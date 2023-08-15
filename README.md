# GMD_market_making
This repo implements the Glosten-Milgrom-Das market making model.

It mainly follows Das's paper (https://cs.gmu.edu/~sanmay/papers/das-qf-rev3.pdf) which constructs an approximate solution to the problem.

--- 
## Three agents interract:

* A market maker setting bid and ask price (of one asset described below) at each time
    * tracks its bids and asks and the exptected value of the asset with a probability distribution of the true value (desrcibed below)
* Noisy traders (trade randomly)
* (noisy) informed traders (trade with inside info about the true asset value)
    * Perfectly informed traders knows the true value
    * Noisy informed traders know the true value up to a gaussian noise $\mathcal{N}(0,\sigma_w^2)$

---

## Asset price

The true value evolves according to a jump process with a probability of jump at each time, and a jump amplitude given by another gaussian $\mathcal{N}(0,\sigma^2)$.

---

## Market maker prior on true value

The market maker initially has a gaussian prior centered at the initial value (he knows it) constructed as follows:

* a discrete vector vec_v of possible values for V
* a discrete vector prior_v corresponding to probabilities of V being equal to vec_v[i]
* Every time traders trade: this is new information for the MM who is aware of both:
    * the probabilstic structure of the true value
    * the proportion of noisy and informed traders
    * the probabilities of buy/sell orders by noisy traders
* He will hence have a function to update the posterior probability once he knows which trade was passed ( he does not know which trader traded, only the proportions of them, so the probability of each being selected)
* the posterior is then the prior for the next iteration and allows the MM to effectively track the true value

---

## Simulation

We iterate through time and at each iteration:

* Randomly select one trader (according to the proportions)
* ask the market maker some bid/ask prices (he tracks the expected value, his own bids and asks, and is informed IF a jump occured, but not its direrction nor amplitude)
* The expected value given the bid and ask is computed
* the trader chosen above trades the asset (long, short or not at all depending on its information set)
* although the simulations does not focus much on pnls, pnls of trader and MM are updated
* the MM updates its posterior distribution on V

---
## How to compute the different quantities

### Bid and ask

Much more details are given in Das's paper but I distill here the main ideas:

* Golsten and Milgrom suggets that
    * $bid = \mathbb{E}[V | \textrm{Sell order received}]$
    * $ask = \mathbb{E}[V | \textrm{Buy order received}]$
    * those can be computed by conditionning to ranges of values for V, below bid, between bid and ask and finally above ask
    * by then conditionning further on the type of trader passing the order, they can be expressed with parameters of the model (not forgetting the noise term of informed traders)
    * The fixed point equation in Pa (ask) or Pb (bid) are solved using a scipy fixed point solver

### Expected value

Regarding the expected value:

* we can condition on the type of order received, not forgetting that we can receive no order at all: 

$\mathbb{E}[V] = \mathbb{E} [V | \textrm{Buy order received}] P(\textrm{Buy order received}) + \mathbb{E} [V | \textrm{Sell order received}] P(\textrm{Sell order received})+ \mathbb{E} [V | \textrm{No order received}] P(\textrm{No order received})$

* where we recognize in the first 2 terms the ask and bid prices

### The posterior

* The posterior is updated by iterating though the values Vi stored in teh vec_v vector and explicitely computing $P(V=V_i | \textrm{trade received})$ where we condition only on the trade received and the type of trader


