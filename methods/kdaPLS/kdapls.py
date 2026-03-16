import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import pairwise as kernels
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator

def kdapls(x:np.ndarray, y:np.ndarray, xs:np.ndarray, xt:np.ndarray, 
           n_components:int, 
           l:list, 
           kernel_params:dict={"type":"rbf","gamma":10}):
    
    assert type(l) is list, "l needs to be a list. If only one number, pass the argument as [l]"
    
    # Get dimensions of arrays and initialize matrices

    (ns, k) = np.shape(xs)
    (n, k) = np.shape(x)
    (nt, k) = np.shape(xt)
    
    Y = y.copy()
    if Y.ndim == 1:        
        Y = Y.reshape(-1,1).copy()   
        
    q = Y.shape[1]
    
    if kernel_params["type"] == "primal":
        m = k
    else:
        m = n

    x_weights_ = np.zeros([m, n_components])
    x_scores_ = np.zeros([n, n_components])
    x_scores_st_ = np.zeros([ns+nt, n_components])
    x_loadings_ = np.zeros([m, n_components])
    x_loadings_st_ = np.zeros([m, n_components])
    y_loadings_ = np.zeros([n_components, q])    
    
    # kdapls elements
 
             
    J = (1/n)*np.ones((n,n))
    H = np.eye(n) - J
    Jst = (1/(ns+nt))*np.ones((ns+nt,ns+nt))
    Hst = np.eye(ns+nt) - Jst
    L1 = np.ones((ns+nt,1))
    L1[ns:,0] = -1
    L = L1@L1.T

    xst = np.vstack((xs,xt)) 
    
       
    if kernel_params["type"] == "rbf":
    
        gamma = kernel_params["gamma"]
        K = kernels.rbf_kernel(x, x, gamma = gamma)
        Kst = kernels.rbf_kernel(xst, x, gamma = gamma)
        
    elif kernel_params["type"] == "linear":
        
        K = x@x.T
        Kst = xst@x.T
        
    elif kernel_params["type"] == "primal":
        
        K = x.copy()
        Kst = xst.copy()
    
    
    # Centering elements
    
    y_mean_ = Y.mean(axis=0)
    centering_ = {}
    
    # Source domain
    centering_[0] = {}
    centering_[0]["n"] = n
    centering_[0]["K"] = K
    centering_[0]["y_mean_"] = y_mean_
    
    # Source-target domain
    centering_[1] = {}
    centering_[1]["n"] = ns+nt    
    centering_[1]["K"] = Kst   
    centering_[1]["y_mean_"] = y_mean_
    
    
    # Centering and final reshaping. L is centered
    
    
    if kernel_params["type"] == "primal":
        K = H@K
        Kst = Hst@Kst
    else:
        K = H@K@H
        Kst = Kst - Kst@J - Jst@Kst + Jst@Kst@J        
    Y = H@Y   
    
        
    # Compute LVs    
    
    for i in range(n_components):
        
        
        if len(l) == 1:
            lA = l[0]
        else:
            lA = l[i]       
        
        
        wM = (K.T@Y@Y.T@K) - lA*(Kst.T@L@Kst)
        wd , wm = eigh(wM)         
        w = wm[:,-1]              
        w.shape = (w.shape[0],1)
        
        # Compute scores and normalize
        t = K@w           
        tst = Kst@w
        t = t / np.linalg.norm(t)
        tst = tst / np.linalg.norm(tst)
        
        # Compute loadings        
        p = K.T@t
        pst = Kst.T@tst
        
        # Regress y on t
        c = t.T@Y

 
        
        # Store w,t,p,c
        x_weights_[:, i] = w.reshape(m)        
        x_scores_[:, i] = t.reshape(n)        
        x_scores_st_[:, i] = tst.reshape(ns+nt)
        x_loadings_[:, i] = p.reshape(m)        
        x_loadings_st_[:, i] = pst.reshape(m)
        y_loadings_[i] = c.reshape(q)        

        # Deflate ? This step is not clear yet.  
        
        # K = K - t@p.T - p@t.T + t@p.T@t@t.T
        # Kst = Kst - Kst@t@t.T - tst@pst.T + tst@pst.T@t@t.T
        
        K = K - t@p.T
        Kst = Kst - tst@pst.T 
        
        Y = Y - (t@c)


    # Calculate regression vector
    coef_ = {}
    coef_[0] = x_weights_@(np.linalg.inv(x_loadings_.T@x_weights_))@y_loadings_
    coef_[1] = x_weights_@(np.linalg.inv(x_loadings_st_.T@x_weights_))@y_loadings_ 

    
    # Residuals    
    x_residuals_ = K    
    x_residuals_st_ = Kst
    y_residuals_ = Y
    

    
    
    return coef_, x_scores_, x_scores_st_, x_weights_, x_loadings_, x_loadings_st_, x_residuals_, x_residuals_st_, y_residuals_, y_loadings_, centering_



class KDAPLSRegression(BaseEstimator):
    
    def __init__(self, xs:np.ndarray, xt:np.ndarray, n_components:int=2,kdict=dict(type="rbf",gamma=0.0001), l:list=[0], target_domain:int=0):
        
        self.xs = xs                              # source domain data
        self.xt = xt                              # target domain data
        self.n_components = n_components          # number of LVs in the model
        self.kdict = kdict                        # kernel parameters
        self.l = l                                # regularization parameter
        self.target_domain = target_domain        # final target domain model
        
    def fit(self, X:np.ndarray, y:np.ndarray):
        
        """
        Fit KDAPLS Model
        
        Parameters
        ----------
        l: list
            Regularization parameter.
            
        target_domain: int
            If multiple target domains are passed, target_domain specifies for which of the target domains
            the model should apply. If target_domain=0, the model applies to the source domain,
            if target_domain=1, the model applies to the concatenations of source and target domain.
        """
        
        self.y = y                                # corresponding labels
        self.x = X                                # labeled x-data (usually x = xs)
        self.n, self.n_feature_in = X.shape       # number of x samples and variables

        coef_, x_scores_, x_scores_st_, x_weights_, x_loadings_, x_loadings_st_, x_residuals_, x_residuals_st_, y_residuals_, y_loadings_, centering_ = kdapls(self.x, self.y, self.xs, self.xt, l=self.l, n_components = self.n_components, kernel_params = self.kdict)
        
        self.coef_ = coef_[self.target_domain]
        self.x_scores_ = x_scores_
        self.x_scores_st_ = x_scores_st_
        self.x_weights_ = x_weights_
        self.x_loadings_ = x_loadings_
        self.x_loadings_st_ = x_loadings_st_
        self.x_residuals_ = x_residuals_
        self.x_residuals_st_ = x_residuals_st_
        self.y_residuals_ = y_residuals_
        self.y_loadings_ = y_loadings_
        self.centering_ = centering_[self.target_domain]
        self.centering_all_ = centering_
        self.coef_all_ = coef_
        
    

    def x_centering_(self,X):
        
        if self.kdict["type"] == "rbf":
            Kt = kernels.rbf_kernel(X, self.x, gamma = self.kdict["gamma"])
        elif self.kdict["type"] == "linear":
            Kt = X@self.x.T
        elif self.kdict["type"] == "primal":
            Kt = X.copy()
            
        if self.kdict["type"] == "primal":
            Kt_c = Kt - self.centering_["K"].mean(axis=0)
        else:
            J = (1/self.n)*np.ones((self.n,self.n))
            Jt = (1/self.centering_["n"])*((np.ones((X.shape[0],1)))@np.ones((1,self.centering_["n"])))
            Kt_c = Kt - Kt@J - Jt@self.centering_["K"] + Jt@self.centering_["K"]@J
        
        
        return Kt_c
    
    def x_centering_all_(self,X, target_domain=0):
        
        if self.kdict["type"] == "rbf":
            Kt = kernels.rbf_kernel(X, self.x, gamma = self.kdict["gamma"])
        elif self.kdict["type"] == "linear":
            Kt = X@self.x.T
        elif self.kdict["type"] == "primal":
            Kt = X.copy()
            
        if self.kdict["type"] == "primal":
            Kt_c = Kt - self.centering_all_[target_domain]["K"].mean(axis=0)
        else:
            J = (1/self.n)*np.ones((self.n,self.n))
            Jt = (1/self.centering_all_[target_domain]["n"])*((np.ones((X.shape[0],1)))@np.ones((1,self.centering_all_[target_domain]["n"])))
            Kt_c = Kt - Kt@J - Jt@self.centering_all_[target_domain]["K"] + Jt@self.centering_all_[target_domain]["K"]@J
        
        
        return Kt_c
        
        
        

    def transform(self, X):
        
        Kt_c = self.x_centering_(X)

        if self.target_domain == 0:
            R = self.x_weights_@(np.linalg.inv(self.x_loadings_.T@self.x_weights_))
        elif self.target_domain == 1:
            R = self.x_weights_@(np.linalg.inv(self.x_loadings_st_.T@self.x_weights_))
        
        Kt_scores = Kt_c@R
        
        return Kt_scores
        
        
        
    def predict(self, X):
        
        Kt_c = self.x_centering_(X)
        Y_pred = Kt_c@self.coef_ + self.centering_["y_mean_"]
        
        return Y_pred
    
    def predict_all(self, X):
        
        Y_pred_ = {}
        for model_i in self.centering_all_.keys():
            Kt_c = self.x_centering_all_(X, model_i)
            Y_pred = Kt_c@self.coef_all_[model_i] + self.centering_all_[model_i]["y_mean_"]
            Y_pred_[model_i] = Y_pred.copy()
            
        return Y_pred_
            
   