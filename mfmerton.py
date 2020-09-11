import numpy as np
import numpy.random as rnd
import math
import time

def datagen(noblgr, nfctr):
    nshadow = nfctr * 3
    ead = rnd.uniform(0, 1, noblgr)
    lgd = rnd.uniform(0, ead)
    s = ead / np.sum(ead)
    rtn = rnd.normal(1, 0.2, noblgr)
    logrtn = np.log(rtn)
    araw = rnd.uniform(0, 1, (nfctr, noblgr))
    a = np.sqrt(araw / np.sum(araw, axis = 0))
    b = rnd.uniform(0.1, 0.9, noblgr)
    xl = rnd.uniform(-1, 1, nfctr)
    xs = rnd.uniform(0, 0.1, nfctr)
    c = rnd.uniform(-2.5, -1.5, noblgr)
    return (ead, lgd, s, xl, xs, a, b, c)

def mcsim(s, lgd, xl, xs, a, b, c, nsmp, ncdf=np.vectorize(lambda x: 0.5*(1+math.erf(x/1.41421356237)))):
    x = rnd.normal(xl, xs, (nsmp, xl.shape[0]))
    u = rnd.uniform(0, 1, (nsmp, b.shape[0]))
    return np.sum(s*lgd*(u<ncdf((c-b*(x@a))/np.sqrt(1-b*b))), axis=1)

if __name__ == '__main__':
    noblgr = 1000
    nfctr = 15
    rnd.seed(180801)
    (ead, lgd, s, xl, xs, a, b, c) = datagen(noblgr, nfctr)

    rnd.seed(200721)
    tic = time.time()
    lsmp = mcsim(s, lgd, xl, xs, a, b, c, 20000)
    qtl = np.quantile(lsmp, np.arange(0.1, 1.0, 0.1))
    toc = time.time()

    print('Quantiles: ', qtl)
    print('Execution time: ', toc - tic)

