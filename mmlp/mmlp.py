#######################################################################################
# Little mlp package for learning python and some neural nets NOW WITH CLASSES YEAH!
# Author: Manuel Hass
# 2017
# 
#######################################################################################
try:    
    import numpy as np
    numpy = np
except ImportError:
    print ('ERROR -> MODULE MISSING: numpy ')


###################### error functions ####################################################
###binary error function (cross entropy)
def bce(ya,yta,dev=False):  ############ work in progress
    if (dev==True):
        return (yta-ya)/((1-yta)*yta)
    return -(np.sum(ya*np.log(yta)+(1.-yta)*np.log(1.-yta))/(yta.shape[0]*2.0))
###Quadratic error function
def qef(ya,yta,dev=False):
    if (dev==True):
        return (yta-ya)
    return np.sum((yta-ya)**2)/(yta.shape[0]*2.0)
###Psudo Huber Loss
def phl(y,yt,dev=False,delta=1.):
    a = (yt-y)
    if (dev==True):
        return  a/( np.sqrt(a**2/delta**2 +1) ) 
    return np.sum((delta**2)*(np.sqrt(1+(a/delta)**2)-1)/(yt.shape[0]*2.0))


###################### regularization ####################################################
### L2 norm
def L2(lam,a):  
    return lam*a
### L1 norm
def L1(lam,a):
    return lam*np.sign(a)


###################### activation functions ####################################################
## robuts logistic transfer fct
def f_lgtr(a,dev=False):
    if (dev==True):
        return (1-np.tanh(a/2.)**2)/2.
    return  (np.tanh(a/2.)+1)/2.
## stochastic transfer fct 
def f_stoch(a,dev=False):
    if (dev==True):
        return np.zeros(a.shape)  
    x = f_lgtr(a,dev=False)
    rand = np.random.random(x.shape)
    return  np.where(rand < x,1,0)
## tanh transfer fct
def f_tanh(a,dev=False):
    if (dev==True):
        return (1-np.tanh(a)**2)
    return  np.tanh(a)
## atan transfer fct
def f_atan(a,dev=False):
    if (dev==True):
        return (1/(a**2+1))
    return  np.arctan(a)
## softplus transfer fct # ~working, make robust
def f_sp(a,dev=False):
    if (dev==True):
        return np.exp(a)/(np.exp(a)+1.)
    return  np.log(np.exp(a)+1.)
## RELU    
def f_relu(a,dev=False):
    if (dev==True):
        return np.maximum(0,np.sign(a)) 
    return  np.maximum(0.0,a)
## Bent ident 
def f_bi(a,dev=False):
    if (dev==True):
         return a / ( 2.0*np.sqrt(a**2+1) ) + 1
    return  (np.sqrt(a**2+1)-1)/2.0 + a
## ident 
def f_iden(a,dev=False):
    if (dev==True):
         return np.ones(a.shape)
    return  a
## binary function
def f_bin(a,dev=False):
    if (dev==True):
         return np.zeros(a.shape) 
    return  np.sign(f_relu(a))


############################# utils ###################################################
def one_hot(targets):
    classes =  np.unique(targets.T)
    binarycoded = []
    for i in classes:
        binarycoded +=  [np.where(targets==i,1,0)[0]]
    return np.array(binarycoded).T

def hot_one(targets):
    return np.argmax(np.array(targets).T,axis=0).reshape(-1,1)
     

################################# MLP ################################    
class mlp:
    def __init__(self,in_dim,h1=32,h2=0,h3=0,out=1):     
        ### initialize weights:
        self.in_dim = in_dim
        self.h1 = h1
        self.h2 = h2
        self.h3 = h3
        self.out = out
        self.count = 0
        # 1-layer
        if (self.h2 == 0) and (self.h3 == 0):
            self.w = np.random.uniform(-1.,1,(self.h1,self.in_dim+1))
            self.v = np.random.uniform(-1.,1,self.h1+1)
            if self.out > 1: self.v = np.random.uniform(-1.,1,(self.out,self.h1+1))
            self.mW = np.random.uniform(0.,1,self.w.shape)
            self.mV = np.random.uniform(0.,1,self.v.shape)
            self.vW = np.random.uniform(0.,1,self.w.shape)
            self.vV = np.random.uniform(0.,1,self.v.shape)
        
        # 2-layer
        elif (self.h2 != 0) and (self.h3 == 0):
            self.w = np.random.uniform(-1.,1,(self.h1,self.in_dim+1))
            self.v = np.random.uniform(-1.,1,(self.h2,self.h1+1))
            self.u = np.random.uniform(-1.,1,self.h2+1) 
            if self.out > 1: self.u = np.random.uniform(-1.,1,(self.out,self.h2+1))
            self.mW = np.random.uniform(0.,1,self.w.shape)
            self.mV = np.random.uniform(0.,1,self.v.shape)
            self.mU = np.random.uniform(0.,1,self.u.shape)
            self.vW = np.random.uniform(0.,1,self.w.shape)
            self.vV = np.random.uniform(0.,1,self.v.shape)
            self.vU = np.random.uniform(0.,1,self.u.shape)
        
        # 3-layer
        else:          
            self.w = np.random.uniform(-1.,1,(self.h1,self.in_dim+1))
            self.v = np.random.uniform(-1.,1,(self.h2,self.h1+1))
            self.u = np.random.uniform(-1.,1,(self.h3,self.h2+1)) 
            self.z = np.random.uniform(-1.,1,self.h3+1)
            if self.out > 1: self.z = np.random.uniform(-1.,1,(self.out,self.h3+1))
            self.mW = np.random.uniform(0.,1,self.w.shape)
            self.mV = np.random.uniform(0.,1,self.v.shape)
            self.mU = np.random.uniform(0.,1,self.u.shape)
            self.mZ = np.random.uniform(0.,1,self.z.shape)
            self.vW = np.random.uniform(0.,1,self.w.shape)
            self.vV = np.random.uniform(0.,1,self.v.shape)
            self.vU = np.random.uniform(0.,1,self.u.shape)
            self.vZ = np.random.uniform(0.,1,self.z.shape)

        ### initialize hyperparameter
        self.f = f_tanh   
        self.f2 = f_iden   
        self.err  = qef   
        self.optimizer = 'Adam'    
        self.beta1 = .85         # smoothing parameter (Adam)
        self.beta2 = .9        # decay rate (RMSprop,Adam)
        self.eta = 1e-2         # initial learning rate
        self.lam = 1e-5         # lambda for L2 weights decay
        self.reg = L2
        self.drop1 = 1.
        self.drop2 = 1.
        self.eps = 1e-7

    def reset(self):
        # 1-layer
        self.count = 0
        if (self.h2 == 0) and (self.h3 == 0):
            self.w = np.random.uniform(-1.,1,(self.h1,self.in_dim+1))
            self.v = np.random.uniform(-1.,1,self.h1+1)
            if self.out > 1: self.v = np.random.uniform(-1.,1,(self.out,self.h1+1))
            self.mW = np.random.uniform(0.,1,self.w.shape)
            self.mV = np.random.uniform(0.,1,self.v.shape)
            self.vW = np.random.uniform(0.,1,self.w.shape)
            self.vV = np.random.uniform(0.,1,self.v.shape)
        
        # 2-layer
        elif (self.h2 != 0) and (self.h3 == 0):
            self.w = np.random.uniform(-1.,1,(self.h1,self.in_dim+1))
            self.v = np.random.uniform(-1.,1,(self.h2,self.h1+1))
            self.u = np.random.uniform(-1.,1,self.h2+1) 
            if self.out > 1: self.u = np.random.uniform(-1.,1,(self.out,self.h2+1))
            self.mW = np.random.uniform(0.,1,self.w.shape)
            self.mV = np.random.uniform(0.,1,self.v.shape)
            self.mU = np.random.uniform(0.,1,self.u.shape)
            self.vW = np.random.uniform(0.,1,self.w.shape)
            self.vV = np.random.uniform(0.,1,self.v.shape)
            self.vU = np.random.uniform(0.,1,self.u.shape)
        
        # 3-layer
        else:          
            self.w = np.random.uniform(-1.,1,(self.h1,self.in_dim+1))
            self.v = np.random.uniform(-1.,1,(self.h2,self.h1+1))
            self.u = np.random.uniform(-1.,1,(self.h3,self.h2+1)) 
            self.z = np.random.uniform(-1.,1,self.h3+1)
            if self.out > 1: self.z = np.random.uniform(-1.,1,(self.out,self.h3+1))
            self.mW = np.random.uniform(0.,1,self.w.shape)
            self.mV = np.random.uniform(0.,1,self.v.shape)
            self.mU = np.random.uniform(0.,1,self.u.shape)
            self.mZ = np.random.uniform(0.,1,self.z.shape)
            self.vW = np.random.uniform(0.,1,self.w.shape)
            self.vV = np.random.uniform(0.,1,self.v.shape)
            self.vU = np.random.uniform(0.,1,self.u.shape)
            self.vZ = np.random.uniform(0.,1,self.z.shape)

    def predict(self,input_):
        x1 = np.vstack((input_.T,np.ones(input_.shape[0]))).T
        w_ = self.drop1*self.w
        v_ = self.drop2*self.v

        if (self.h2 == 0) and (self.h3 == 0):
            H = (np.dot(x1,w_.T)).T
            s = self.f(H)
            self.s1 = np.vstack((s,np.ones(s.shape[1])))
            y__ = self.f2(np.dot(self.s1.T,v_.T))

        elif (self.h2 != 0) and (self.h3 == 0):
            u_ = self.drop2*self.u
            H = (np.dot(x1,w_.T)).T
            s = self.f(H)
            self.s1 = np.vstack((s,np.ones(s.shape[1])))
            H2 = (np.dot(self.s1.T,v_.T))
            self.s2 = self.f(H2.T)
            self.s2 = np.vstack((self.s2,np.ones(self.s2.shape[1]).T))
            H3 = np.dot(self.s2.T,u_.T)
            y__ = self.f2(H3)

        else:
            u_ = self.drop2*self.u
            z_ = self.drop2*self.z
            H = (np.dot(x1,w_.T)).T
            s = self.f(H)
            self.s1 = np.vstack((s,np.ones(s.shape[1])))
            H2 = (np.dot(self.s1.T,v_.T))
            self.s2 = self.f(H2.T)
            self.s2 = np.vstack((self.s2,np.ones(self.s2.shape[1]).T))
            H3 = np.dot(self.s2.T,u_.T)
            self.s3 = self.f(H3)
            self.s3 = np.vstack((self.s3.T,np.ones(self.s3.T.shape[1]).T))
            H4 = np.dot(self.s3.T,z_.T)
            y__ = (self.f2(H4))

        return y__ 
            
           
            

    def train(self,input_,y):
        x1 = np.vstack((input_.T,np.ones(input_.shape[0]))).T
        if self.drop1 != 1.:
            mask0 = np.random.choice([0, 1], size=(x1.shape[1],), p=[1-self.drop1, self.drop1])
        else:
            mask0 = 1.
        if self.drop2 != 1.:
            mask1 = np.random.choice([0, 1], size=(self.h1,), p=[1-self.drop2, self.drop2])
            mask2 = np.random.choice([0, 1], size=(self.h2,), p=[1-self.drop2, self.drop2])
            mask3 = np.random.choice([0, 1], size=(self.h3,), p=[1-self.drop2, self.drop2])
        else:
            mask1 = 1.
            mask2 = 1.
            mask3 = 1.

        # 1-layer
        if (self.h2 == 0) and (self.h3 == 0):
            #forwards
            H =  (np.dot(mask0 *x1,self.w.T)).T
            s = (mask1 * self.f(H).T).T
            self.s1 = np.vstack((s,np.ones(s.shape[1])))
            H1 = np.dot(self.s1.T,self.v.T)
            self.y_ = (self.f2(H1))
            #backwards
            if self.out > 1:
                dv = (self.err(y,self.y_,dev=True)*self.f2(H1,True)).T
                self.dV = -1.0/x1.shape[0] * (np.dot(dv,self.s1.T)) - self.reg(self.lam,self.v)
                dw = np.dot(self.v.T[1:],dv)*(mask1*self.f(H,True).T).T 
                self.dW = -1.0/x1.shape[0] * (np.dot(dw,mask0*x1)) - self.reg(self.lam,self.w)
            else:
                self.dV = -1.0/x1.shape[0] * np.dot(self.s1,self.err(y,self.y_,dev=True).T) - self.reg(self.lam,self.v)
                self.dW = -1.0/x1.shape[0] * np.dot(np.diag(self.v[1:]),(mask1*self.f(H,True).T).T).dot(np.diag(self.err(y,self.y_,dev=True))).dot((mask0*x1)) - self.reg(self.lam,self.w)
        
        # 2-layer
        elif (self.h2 != 0) and (self.h3 == 0):
            #forwards
            H = (np.dot(x1,self.w.T)).T
            s = self.f(H)
            self.s1 = np.vstack((s,np.ones(s.shape[1])))
            H2 = (np.dot(self.s1.T,self.v.T))
            self.s2 = self.f(H2.T)
            self.s2 = np.vstack((self.s2,np.ones(self.s2.shape[1]).T))
            H3 = np.dot(self.s2.T,self.u.T)
            self.y_ = self.f2(H3)
            #backwards
            if self.out > 1:
                du = (self.err(y,self.y_,dev=True)*self.f2(H3,True)).T
                self.dU = -1.0/x1.shape[0] * (np.dot(du,self.s2.T)) - self.reg(self.lam,self.u)
                dv = np.dot(self.u.T[1:],du)*(mask2*self.f(H2,True)).T
                self.dV = -1.0/x1.shape[0] * (np.dot(dv,self.s1.T)) - self.reg(self.lam,self.v)
                dw = np.dot(self.v.T[1:],dv)*(mask1*self.f(H,True).T).T 
                self.dW = -1.0/x1.shape[0] * (np.dot(dw,mask0*x1)) - self.reg(self.lam,self.w)
            else:
                self.dU = -1.0/x1.shape[0] * (np.dot(self.err(y,self.y_,dev=True),self.s2.T)) - self.reg(self.lam,self.u)
                du = np.diag(self.err(y,self.y_,dev=True))## np.diag(np.dot(np.diag(z[1:]),self.f(H3.T,True)).dot(dz).T.sum(axis=1))
                self.dV = -1.0/x1.shape[0] * np.dot(np.diag(self.u[1:]),(mask2*self.f(H2.T,True).T).T).dot(du).dot(self.s1.T) - self.reg(self.lam,self.v)
                dv = np.diag(np.dot(np.diag(self.u[1:]),(mask2*self.f(H2.T,True).T).T).dot(du).T.sum(axis=1))     
                self.dW = -1.0/x1.shape[0] * np.dot(np.diag(self.v.T[1:].sum(axis=1)),(mask1*self.f(H,True).T).T).dot(dv).dot(mask0*x1) - self.reg(self.lam,self.w)

        # 3-layer
        else:
            #forwards
            H = (np.dot(mask0*x1,self.w.T)).T
            s = (mask1*self.f(H).T).T
            self.s1 = np.vstack((s,np.ones(s.shape[1])))
            H2 = (np.dot(self.s1.T,self.v.T))
            self.s2 = (mask2*self.f(H2.T).T).T
            self.s2 = np.vstack((self.s2,np.ones(self.s2.shape[1]).T))
            H3 = np.dot(self.s2.T,self.u.T)
            self.s3 = (mask3*self.f(H3))
            self.s3 = np.vstack((self.s3.T,np.ones(self.s3.T.shape[1]).T))
            H4 = np.dot(self.s3.T,self.z.T)
            self.y_ = (self.f2(H4))
            #backwards
            if self.out > 1:
                dz = (self.err(y,self.y_,dev=True)*self.f2(H4,True)).T
                self.dZ = -1.0/x1.shape[0] * (np.dot(dz,self.s3.T)) - self.reg(self.lam,self.z)
                du = np.dot(self.z.T[1:],dz)*(mask3*self.f(H3,True)).T
                self.dU = -1.0/x1.shape[0] * (np.dot(du,self.s2.T)) - self.reg(self.lam,self.u)
                dv = np.dot(self.u.T[1:],du)*(mask2*self.f(H2,True)).T
                self.dV = -1.0/x1.shape[0] * (np.dot(dv,self.s1.T)) - self.reg(self.lam,self.v)
                dw = np.dot(self.v.T[1:],dv)*(mask1*self.f(H,True).T).T 
                self.dW = -1.0/x1.shape[0] * (np.dot(dw,mask0*x1)) - self.reg(self.lam,self.w)
            else:
                self.dZ = -1.0/x1.shape[0] * (np.dot(self.err(y,self.y_,dev=True),self.s3.T)) - self.reg(self.lam,self.z)
                dz = np.diag(self.err(y,self.y_,dev=True))
                self.dU = -1.0/x1.shape[0] * np.dot(np.diag(self.z[1:]),(mask3*self.f(H3.T,True).T).T).dot(dz).dot(self.s2.T) - self.reg(self.lam,self.u)
                du = np.diag(np.dot(np.diag(self.z[1:]),(mask3*self.f(H3.T,True).T).T).dot(dz).T.sum(axis=1))
                self.dV = -1.0/x1.shape[0] * np.dot(np.diag(self.u.T[1:].sum(axis=1)),(mask2*self.f(H2.T,True).T).T).dot(du).dot(self.s1.T) - self.reg(self.lam,self.v)
                dv = np.diag(np.dot(np.diag(self.u.T[1:].sum(axis=1)),(mask2*self.f(H2.T,True).T).T).dot(du).T.sum(axis=1))
                self.dW = -1.0/x1.shape[0] * np.dot(np.diag(self.v.T[1:].sum(axis=1)),(mask1*self.f(H,True).T).T).dot(dv).dot(mask0*x1) - self.reg(self.lam,self.w)

        self.mW = self.beta1*self.mW + (1-self.beta1)*self.dW
        self.mV = self.beta1*self.mV + (1-self.beta1)*self.dV
        self.vW = self.beta2*self.vW + (1-self.beta2)* self.dW**2 
        self.vV = self.beta2*self.vV + (1-self.beta2)* self.dV**2 
        if self.h2 > 0:
            self.mU = self.beta1*self.mU + (1-self.beta1)*self.dU
            self.vU =  self.beta2*self.vU + (1-self.beta2)* self.dU**2
        if self.h3 > 0:
            self.vZ =  self.beta2*self.vZ + (1-self.beta2)* self.dZ**2   
            self.mZ = self.beta1*self.mZ + (1-self.beta1)*self.dZ

        if(self.optimizer=='RMSprop'):
            self.w += self.eta* self.dW / (np.sqrt(self.vW) +self.eps)
            self.v += self.eta* self.dV / (np.sqrt(self.vV) +self.eps)
            if self.h2 > 0:
                self.u += self.eta* self.dU / (np.sqrt(self.vU) +self.eps)
            if self.h3 > 0:
                self.z += self.eta* self.dZ / (np.sqrt(self.vZ) +self.eps)
        if(self.optimizer=='normal'):
            self.w += self.eta* self.dW 
            self.v += self.eta* self.dV 
            if self.h2 > 0:
                self.u += self.eta* self.dU 
            if self.h3 > 0:
                self.z += self.eta* self.dZ 
        if (self.optimizer=='Adam'):
            self.w += self.eta* self.mW / (np.sqrt(self.vW) +self.eps)
            self.v += self.eta* self.mV / (np.sqrt(self.vV) +self.eps)
            if self.h2 > 0:
                self.u += self.eta* self.mU / (np.sqrt(self.vU) +self.eps)
            if self.h3 > 0:
                self.z += self.eta* self.mZ / (np.sqrt(self.vZ) +self.eps)

        self.loss = self.err(y,self.y_)
        self.count += 1