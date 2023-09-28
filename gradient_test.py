
import numpy as np
#定義函數 f(x) 和它的導數 df(x):
f = lambda x: (1/16)*x[0]**2+9*x[1]**2
df = lambda x: np.array( ((1/8)*x[0],18*x[1]))
#使用中央差分方法，定義數值逼近的導數 df_approx:
df_approx = lambda x,eps:((f([x[0]+eps,x[1]])-f([x[0]-eps,x[1]]) )/(2*eps),( f([x[0],x[1]+eps])-f([x[0],x[1]-eps]) )/(2*eps))
#為 x 和 eps 賦值，計算 df(x) 和 df_approx(x, eps)，然後輸出其值和它們之間的差異:
x = [2.,3.]
eps = 1e-8
grad = df(x)
grad_approx = df_approx(x,eps)
print(grad)
print(grad_approx)
print(abs(grad-grad_approx))
#定義 numerical_gradient 函數，用於計算任意多維函數的數值梯度:
def numerical_gradient(f,params,eps = 1e-6):
    numerical_grads = []
    for x in params:
        # x可能是多維數組，對其每個元素，計算其數值偏導數
        grad = np.zeros(x.shape)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) #np.nditer() 用於遍歷每一個元素並計算其數值梯度。
        while not it.finished:            
            idx = it.multi_index
            old_value = x[idx]        
            x[idx] = old_value + eps  # x[idx]+eps
            fx = f()  
            x[idx] = old_value - eps #  x[idx]-eps
            fx_ = f()
            grad[idx] = (fx - fx_) / (2*eps)  
            x[idx] = old_value      #注意：一定要將該權值參數還原到原來的值。
            it.iternext()           # 循環存取x的下一個元素
      
        numerical_grads.append(grad)
    return numerical_grads

#設定一組新的輸入值 x 和參數 param，然後使用 numerical_gradient 函數來計算其梯度:
x = np.array([2.,3.])
param = np.array(x)        #numerical_gradient的参数param必须是numpy数组
numerical_grads = numerical_gradient(lambda:f(param),[param],1e-6)
print(numerical_grads[0])
#使用另一種方式來計算 x 的梯度:
def fun():
    return f(param)

numerical_grads = numerical_gradient(fun,[param],1e-6)
print(numerical_grads[0])