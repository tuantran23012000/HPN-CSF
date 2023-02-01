import numpy as np

from epo_lp import EPO_LP
from scipy.optimize import minimize
from scipy.optimize import Bounds,BFGS
from autograd import grad
def g3(x):
    return x[0]**2 + x[1]**2 + x[2]**2 - 1
g3_dx = grad(g3)
bounds = Bounds([0 for i in range(1)], [1 for i in range(1)])
cons = ({'type': 'eq',
          'fun' : lambda x: np.array([g3(x)]),
          'jac' : lambda x: np.array([g3_dx(x)])})
def rosen(x,y):
    """The Rosenbrock function"""
    return np.sqrt(np.sum((x-y)**2))
def find_min(y,n):
    x = np.random.rand(1,n).tolist()[0]
    res = minimize(rosen, x, args=(y), jac="2-point",hess = BFGS(),
                method='trust-constr', options={'disp': False},bounds=bounds)
    return res.x


# sol = find_min(x_init)
def epo_search(multi_obj_fg, r, x=None, relax=False, eps=1e-4, max_iters=100,
               n_dim=20, step_size=.1, grad_tol=1e-4, store_xs=False):
    if relax:
        print('relaxing')
    else:
        print('Restricted')
    # randomly generate one solution
    x = np.random.randn(n_dim) if x is None else x
    m = len(r)       # number of objectives
    lp = EPO_LP(m, n_dim, r, eps=eps)
    ls, mus, adjs, gammas, lambdas = [], [], [], [], []
    if store_xs:
        xs = [x]

    # find the Pareto optimal solution
    desc, asce = 0, 0
    for t in range(max_iters):
        l, G = multi_obj_fg(x)
        #print(l,G)
        alpha = lp.get_alpha(l, G, relax=relax)
        if lp.last_move == "dom":
            desc += 1
        else:
            asce += 1
        ls.append(l)
        lambdas.append(np.min(r * l))
        mus.append(lp.mu_rl)
        adjs.append(lp.a.value)
        gammas.append(lp.gamma)

        d_nd = alpha @ G
        if np.linalg.norm(d_nd, ord=np.inf) < grad_tol:
            print('converged, ', end=',')
            break
        x = x - 10. * max(lp.mu_rl, 0.1) * step_size * d_nd
        #print(x)
        x = find_min(x[0],1)
        
        #print(type(x))
        if store_xs:
            xs.append(x)
        x = np.array([x])
        #print(x)

    print(f'# iterations={asce+desc}; {100. * desc/(desc+asce)} % descent')
    res = {'ls': np.stack(ls),
           'mus': np.stack(mus),
           'adjs': np.stack(adjs),
           'gammas': np.stack(gammas),
           'lambdas': np.stack(lambdas)}
    if store_xs:
        res['xs': xs]
    return x, res
