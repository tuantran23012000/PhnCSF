import numpy as np
class PB2():
    def __init__(self):
        super().__init__()

    def f_1(self,x):
        return (1/50)*(x[0][0]**2 + x[0][1]**2)
    def f_2(self,x):
        return (1/50)*((x[0][0]-5)**2 + (x[0][1]-5)**2)
    def evaluate(self,x):
        return np.stack([self.f_1(x),self.f_2(x)])
    def create_pf(self,x):
        ps = np.linspace(0,5,num = 1000)
        pf = []
        for x1 in ps:
            x = np.array([[x1,x1]])
            f= self.evaluate(x)
            pf.append(f)   
        pf = np.array(pf)
        return pf