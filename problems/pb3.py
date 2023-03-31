import numpy as np
class PB3():
    def __init__(self):
        super().__init__()

    def f_1(self,x):
        return ((x[0][0]**2 + x[0][1]**2 + x[0][2]**2+x[0][1] - 12*(x[0][2])) +12)/1
    def f_2(self,x):
        return ((x[0][0]**2 + x[0][1]**2 + x[0][2]**2\
        + 8*(x[0][0]) - 44.8*(x[0][1]) + 8*(x[0][2])) +44)/57
    def f_3(self,x):
        return ((x[0][0]**2 + x[0][1]**2 + x[0][2]**2 -44.8*(x[0][0])\
            + 8*(x[0][1]) + 8*(x[0][2]))+43.7)/56
    def evaluate(self,x):
        return np.stack([self.f_1(x),self.f_2(x),self.f_3(x)])
    def create_pf(self,x):
        u = np.linspace(0, 1, endpoint=True, num=60)
        v = np.linspace(0, 1, endpoint=True, num=60)
        tmp = []
        for i in u:
            for j in v:
                if 1-i**2-j**2 >=0:
                    tmp.append([np.sqrt(1-i**2-j**2),i,j])
                    tmp.append([i,np.sqrt(1-i**2-j**2),j])
                    tmp.append([i,j,np.sqrt(1-i**2-j**2)])
        uv = np.array(tmp)
        print(f"uv.shape={uv.shape}")
        ls = []
        for x in uv:
            x = np.array([x])
            f= self.concave_fun_eval_3d(x)
            ls.append(f)
        ls = np.stack(ls)
        po, pf = [], []
        for i, x in enumerate(uv):
            l_i = ls[i]
            po.append(x)
            pf.append(l_i)
        po = np.stack(po)
        pf = np.stack(pf)
        return pf