import numpy as np

from min_norm_solvers_numpy import MinNormSolver
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
# def epo_search(multi_obj_fg, r, x=None, relax=False, eps=1e-4, max_iters=100,
#                n_dim=20, step_size=.1, grad_tol=1e-4, store_xs=False):
#     if relax:
#         print('relaxing')
#     else:
#         print('Restricted')
#     # randomly generate one solution
#     x = np.random.randn(n_dim) if x is None else x
#     m = len(r)       # number of objectives
#     lp = EPO_LP(m, n_dim, r, eps=eps)
#     ls, mus, adjs, gammas, lambdas = [], [], [], [], []
#     if store_xs:
#         xs = [x]

#     # find the Pareto optimal solution
#     desc, asce = 0, 0
#     for t in range(max_iters):
#         l, G = multi_obj_fg(x)
#         #print(l,G)
#         alpha = lp.get_alpha(l, G, relax=relax)
#         if lp.last_move == "dom":
#             desc += 1
#         else:
#             asce += 1
#         ls.append(l)
#         lambdas.append(np.min(r * l))
#         mus.append(lp.mu_rl)
#         adjs.append(lp.a.value)
#         gammas.append(lp.gamma)

#         d_nd = alpha @ G
#         if np.linalg.norm(d_nd, ord=np.inf) < grad_tol:
#             print('converged, ', end=',')
#             break
#         x = x - 10. * max(lp.mu_rl, 0.1) * step_size * d_nd
#         #print(x)
#         x = find_min(x[0],3)
        
#         #print(type(x))
#         if store_xs:
#             xs.append(x)
#         x = np.array([x])
#         #print(x)

#     print(f'# iterations={asce+desc}; {100. * desc/(desc+asce)} % descent')
#     res = {'ls': np.stack(ls),
#            'mus': np.stack(mus),
#            'adjs': np.stack(adjs),
#            'gammas': np.stack(gammas),
#            'lambdas': np.stack(lambdas)}
#     if store_xs:
#         res['xs': xs]
#     return x, res
def get_d_paretomtl_init(grads,value,weights,i):
    # calculate the gradient direction for Pareto MTL initialization
    nobj, dim = grads.shape
    
    # check active constraints
    normalized_current_weight = weights[i]/np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, (i), axis=0) / np.linalg.norm(np.delete(weights, (i), axis=0), axis = 1,keepdims = True)
    w = normalized_rest_weights - normalized_current_weight
    
    gx =  np.dot(w,value/np.linalg.norm(value))
    idx = gx >  0
    
    if np.sum(idx) <= 0:
        return np.zeros(nobj)
    if np.sum(idx) == 1:
        sol = np.ones(1)
    else:
        vec =  np.dot(w[idx],grads)
        sol, nd = MinNormSolver.find_min_norm_element(vec)

    # calculate the weights
    weight0 =  np.sum(np.array([sol[j] * w[idx][j ,0] for j in np.arange(0, np.sum(idx))]))
    weight1 =  np.sum(np.array([sol[j] * w[idx][j ,1] for j in np.arange(0, np.sum(idx))]))
    weight = np.stack([weight0,weight1])
   

    return weight
def get_d_paretomtl(grads,value,weights,i):
    # calculate the gradient direction for Pareto MTL
    nobj, dim = grads.shape
    
    # check active constraints
    normalized_current_weight = weights[i]/np.linalg.norm(weights[i])
    normalized_rest_weights = np.delete(weights, (i), axis=0) / np.linalg.norm(np.delete(weights, (i), axis=0), axis = 1,keepdims = True)
    w = normalized_rest_weights - normalized_current_weight
    
    
    # solve QP 
    gx =  np.dot(w,value/np.linalg.norm(value))
    idx = gx >  0
   
    
    vec =  np.concatenate((grads, np.dot(w[idx],grads)), axis = 0)
    
    #    # use cvxopt to solve QP
    #    
    #    P = np.dot(vec , vec.T)
    #    
    #    q = np.zeros(nobj + np.sum(idx))
    #    
    #    G =  - np.eye(nobj + np.sum(idx) )
    #    h = np.zeros(nobj + np.sum(idx))
    #    
    #
    #    
    #    A = np.ones(nobj + np.sum(idx)).reshape(1,nobj + np.sum(idx))
    #    b = np.ones(1)
    
    #    cvxopt.solvers.options['show_progress'] = False
    #    sol = cvxopt_solve_qp(P, q, G, h, A, b)
  
    # use MinNormSolver to solve QP
    sol, nd = MinNormSolver.find_min_norm_element(vec)
   
    
    # reformulate ParetoMTL as linear scalarization method, return the weights
    weight0 =  sol[0] + np.sum(np.array([sol[j] * w[idx][j - 2,0] for j in np.arange(2,2 + np.sum(idx))]))
    weight1 = sol[1] + np.sum(np.array([sol[j] * w[idx][j - 2,1] for j in np.arange(2,2 + np.sum(idx))]))
    weight = np.stack([weight0,weight1])
   

    return weight
def pareto_mtl_search(multi_obj_fg,ref_vecs,x,i,t_iter = 1000, step_size = 1):
    """
    Pareto MTL
    """

    # randomly generate one solution
    #x = np.random.uniform(-0.5,0.5,n_dim)
    sols = []
    l = []
    # find the initial solution
    #print(x.shape)
    for t in range(int(t_iter * 0.2)):
        #print(t)
        f, f_dx = multi_obj_fg(x)
        #print(f,f_dx)
        #print(ref_vecs,i)
        #ref_vecs = np.array(ref_vecs)
        weights =  get_d_paretomtl_init(f_dx,f,ref_vecs,i)
        #print(weights.shape)
        
        x = x - step_size * np.dot(weights.T,f_dx).flatten()
        # x = find_min(x[0],1)
        # x = np.array([x])
        l.append(f)
        sols.append(x)
    # find the Pareto optimal solution
    
    for t in range(int(t_iter * 0.8)):
        f, f_dx = multi_obj_fg(x)
        #print(f,f_dx)
        weights =  get_d_paretomtl(f_dx,f,ref_vecs,i)
        #print(weights)
        
        x = x - step_size * np.dot(weights.T,f_dx).flatten()
        # x = find_min(x[0],1)
        # x = np.array([x])
        l.append(f)
        sols.append(x)
    
    return sols,l
