import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(df,x,alpha=0.01, iterations = 100,epsilon = 1e-8):
    history=[x]
    for i in range(iterations):
        if abs(df(x))<epsilon:#看gradient是否夠小了
            print("gradient低於設定值了")
            break
        x = x-alpha* df(x)        
        history.append(x)
    return history


#畫一條曲線
f = lambda x: np.power(x,3)-3*x**2-9*x+2
x = np.arange(-3, 4, 0.01)
y= f(x)
plt.plot(x,y)



df = lambda x: 3*x**2-6*x-9#上面曲線的微分
path = gradient_descent(df,1.,0.01,200)#從x=1的點開始計算gradient，並且一直找，直到找到gradient接近零的位置
print(path[-1])




path_x = np.asarray(path)#轉成np
path_y=f(path_x)
#畫箭頭
plt.quiver(path_x[:-1], path_y[:-1], path_x[1:]-path_x[:-1], path_y[1:]-path_y[:-1], scale_units='xy', angles='xy', scale=1, color='k')
#畫點
plt.scatter(path[-1],f(path[-1]))
plt.show()