import numpy as np
class PB1():
    def __init__(self):
        super().__init__()

    def f_1(self,x):
        return x[0][0]
    def f_2(self,x):
        return (x[0][0]-1)**2
    def evaluate(self,x):
        return np.stack([self.f_1(x),self.f_2(x)])
    def create_pf(self,x):
        ps = np.linspace(0,1,num = 1000)
        pf = []
        for x1 in ps:
            x = np.array([[x1]])
            f= self.evaluate(x)
            pf.append(f)   
        pf = np.array(pf)
        return pf