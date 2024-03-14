import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchvision

# unimodal and bimodal normal distributions
def unimodal_distribution(x, mu, sigma):
    return 1/(torch.sqrt(2 * torch.pi * sigma ** 2)) * torch.exp(-(x - mu)**2 / (2 * sigma**2))

def bimodal_distribution(x, mu1, sigma1, mu2, sigma2, alpha):
    # alpha is the weight of each gaussian
    return alpha * unimodal_distribution(x, mu1, sigma1) + (1 - alpha) * unimodal_distribution(x, mu2, sigma2)

def kl_divergence(p, q):
    return p * (torch.log(p) - torch.log(q))

def simplex(weights):
    n, = weights.shape
    # get the array of cumulative sums of a sorted (decreasing) copy of weights
    sorted_weights = torch.stack(torch.sort(weights)[::-1])
    cum_sum = torch.cumsum(sorted_weights, dim=0)
    # get the number of > 0 components of the optimal solution
    nonz = torch.nonzero(sorted_weights * torch.arange(1, n + 1) > (cum_sum - 1))
    lam = nonz[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cum_sum[lam] - 1) / (lam + 1.0)
    updated_weights = (weights - theta).clip(min=0)
    return updated_weights

def vi_fkl(x, p, boosting_iterations=20, iterations=400):
    # we keep the qi's in memory
    qi = []
    weights = []
    for boosting in range(boosting_iterations):
        mu = torch.zeros(1)
        if(boosting > 0): # set mu to the highest remainder index
            remainder = (p / fi)
            mu = x[remainder.argmax()].clone()
        mu.requires_grad = True
        sigma = torch.tensor(1., requires_grad = True)
        
        lmbd = torch.tensor(0.5, requires_grad = True) # the 0.5 comes from Miller's paper (they use EM afterwards)
        if(boosting == 0):
            lmbd = torch.tensor(1.) # if no previous component was fitted then it is naturally 1
        weights += [lmbd]
        
        optimizer = optim.Adam([mu, sigma, lmbd], lr = 0.1)
        
        for iteration in range(iterations):
            
            # define the unimodal distribution
            q = unimodal_distribution(x, mu, sigma)
            
            if(boosting > 0):
                if(iteration > 20000): # it seems that updating gamma along the iteration is not beneficial, this condition is never satisfied
                    q = lmbd * q + (1 - lmbd) * fi
                else:
                    q = 0.5 * q + (1 - 0.5) * fi # if we have more than 1 component, q is a mixture of gaussians
            
            # calculate KL divergence
            p = torch.clamp(p, min = 1e-8) # common trick to avoid infinity
            q = torch.clamp(q, min = 1e-8)
            loss = kl_divergence(p, q) # forward KL
            if(iteration == 0):
                # initialization with RKL
                loss = kl_divergence(q, p) # reversed KL
            elif(boosting > 0):
                # importance sampling
                w = p / fi # equation (22) of the paper
                w = w / w.sum() # intuitively, we would give more weights to values where f_i is largely different from p
                loss = w * loss
            loss = loss.mean()
            
            # perform gradient descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # we add q into the qi's tab
        qi += [unimodal_distribution(x, mu, sigma).clone().clamp(min = 1e-8)]
        mu.requires_grad = False
        sigma.requires_grad = False
        
        # fully re-optimized using simplex-projected gradient descent
        weights = torch.stack(weights)
        for it in range(100):
            numerator = torch.stack(qi) * p # equation (23)
            q_sum = (weights[:, None] * torch.stack(qi)).sum(axis = 0) # denominator
            grad = (numerator / q_sum).sum(axis = -1) # expectation
            weights -= 0.001 * grad # update weights
            weights = torch.tensor(simplex(weights.clone().detach()))
        
        # weights are updated according to: "To ensure non-negative or zero mixture weights, we optimize
        # over the logits of the weights from which we recover the final weights by logistic transformation" appendix E.1
        weights = torch.softmax(weights, dim = 0)
        fi = (torch.stack(qi) * weights[:, None]).sum(axis = 0).detach()
        weights = list(weights.detach())
    return fi, qi

def imshow(img):
    pil_img = torchvision.transforms.functional.to_pil_image(img)
    display(pil_img)
    return(pil_img)