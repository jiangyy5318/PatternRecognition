import numpy as np
import math

def IndexIMI(s,t):
    print s
    print t
    if len(s) != len(t):
        print 'Error,Different length between predict and label'
        return -1
    len_s = max(s)
    len_t = max(t)
    Cnt = np.zeros((len_s+1,len_t+1))
    for e,f in zip(s,t):
        Cnt[e,f] = Cnt[e,f] + 1

    sum_s = Cnt.sum(axis=0)
    sum_t = Cnt.sum(axis=1)
    total = Cnt.sum()
    Numerator = 0.0
    for i in range(len_s+1):
        for j in range(len_t+1):
            if Cnt[i, j] == 0:
                continue
            Numerator = Numerator + Cnt[i, j] * math.log(float(total) * Cnt[i, j] / sum_s[j] / sum_t[i])
    # Numerator = sum(x*math.log(float(x)*TotalSum/m,n)   for )
    Denominator1 = sum(x * math.log(float(x) / total) for x in sum_s if x != 0)
    Denominator2 = sum(x * math.log(float(x) / total) for x in sum_t if x != 0)
    return Numerator / math.sqrt(Denominator1 * Denominator2)


if __name__ == '__main__':
     print IndexIMI([0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2])
     print IndexIMI([0, 0, 1, 1,2,2], [2,2, 0, 0,1, 1])