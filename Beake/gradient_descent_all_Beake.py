

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


#Beake's function
f  = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

#定義函數的最小值點:
minima = np.array([3., .5]) #size(2,) 
minima_ = minima.reshape(-1, 1) #size(2,1)
#建立格點座標:
xmin, xmax, xstep = -4.5, 4.5, .2
ymin, ymax, ystep = -4.5, 4.5, .2
x_list = np.arange(xmin, xmax + xstep, xstep) #(46,)
y_list = np.arange(ymin, ymax + ystep, ystep) #(46,)
x, y = np.meshgrid(x_list, y_list)
#計算函數值:
z = f(x, y)
#繪製三維曲面圖:
fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=50, azim=-50)

ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, 
                edgecolor='none', alpha=.8, cmap=plt.cm.jet)
#norm=LogNorm()將色彩映射到對數刻度上
#rstride=1, cstride=1每一行和每一列都被繪製，沒有略過任何格子。
#edgecolor='none'每個格子的邊緣顏色
#alpha=.8透明度
#cmap=plt.cm.jet色彩映射，從藍色到紅色的漸變。
#繪製最小值點(紅色的星形)(星形的大小)
ax.plot(*minima_, f(*minima_), 'r*', markersize=10) 
#也等於ax.plot(3., 0.5, f(3., 0.5), 'r*', markersize=10)

#設定軸標籤和範圍:
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
#顯示圖片
plt.show()

#定義 Beale's function 的偏導數( 相對於 x 和 y 的偏導數。)
df_x  = lambda x, y: 2*(1.5 - x + x*y)*(y-1) + 2*(2.25 - x + x*y**2)*(y**2-1) + 2*(2.625 - x + x*y**3)*(y**3-1)
df_y  = lambda x, y: 2*(1.5 - x + x*y)*x + 2*(2.25 - x + x*y**2)*(2*x*y) + 2*(2.625 - x + x*y**3)*(3*x*y**2)
#計算偏導數:
dz_dx = df_x(x, y)
dz_dy = df_y(x, y)
#創建畫布和軸:
fig, ax = plt.subplots(figsize=(10, 6))
#繪製等高線圖:

ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet)
#levels 參數用於定義等高線的高度
#np.logspace(0, 5, 35)生成35個在 [10**0 (也就是1), 10**5] 範圍內的數字
#norm=LogNorm() 使得等高線在對數刻度上均勻分布。

#繪製gradient向量:
ax.quiver(x, y, x - dz_dx, y - dz_dy, alpha=.5)

#繪製最小值點:
ax.plot(*minima_, 'r*', markersize=18)
#設定軸標籤和範圍:
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
#顯示圖形:
plt.show()

df = lambda x: np.array( [2*(1.5 - x[0] + x[0]*x[1])*(x[1]-1) + 2*(2.25 - x[0] + x[0]*x[1]**2)*(x[1]**2-1)
                                        + 2*(2.625 - x[0] + x[0]*x[1]**3)*(x[1]**3-1),
                           2*(1.5 - x[0] + x[0]*x[1])*x[0] + 2*(2.25 - x[0] + x[0]*x[1]**2)*(2*x[0]*x[1]) 
                                         + 2*(2.625 - x[0] + x[0]*x[1]**3)*(3*x[0]*x[1]**2)])
def gradient_descent(df,x,alpha=0.01, iterations = 100,epsilon = 1e-8):
    history=[x]
    for i in range(iterations):
        if np.max(np.abs(df(x)))<epsilon:
            print("gradient低於設定值了")
            break
        x = x-alpha* df(x)       
        history.append(x)
    return history
def gradient_descent_momentum(df,x,alpha=0.01,gamma = 0.8, iterations = 100,epsilon = 1e-6):
    history=[x]
    v= np.zeros_like(x)            #初始v等於0
    for i in range(iterations):
        if np.max(np.abs(df(x)))<epsilon:
            print("gradient足够小！")
            break
        v = gamma*v+alpha* df(x)  #更新动量
        x = x-v                   #更新变量（参数）
        
        history.append(x)
    return history
def gradient_descent_Adagrad(df,x,alpha=0.01,iterations = 100,epsilon = 1e-8):
    history=[x]
    #gl= np.zeros_like(x) #初始值
    gl = np.ones_like(x) * 0.1
    for i in range(iterations):
        if np.max(np.abs(df(x)))<epsilon:
            print("gradient足够小！")
            break
        grad = df(x)
        gl += grad**2
        x = x-alpha* grad/(np.sqrt(gl)+epsilon)      
        history.append(x)
    return history
def gradient_descent_Adadelta(df, x, rho=0.95, iterations=100, epsilon=1e-8):
    history = [x]
    Eg = np.zeros_like(x)
    Edelta = np.zeros_like(x)
    for i in range(iterations):
        if np.max(np.abs(df(x))) < epsilon:
            print("gradient足够小！")
            break
        grad = df(x)
        Eg = rho * Eg + (1 - rho) * (grad**2) #先計算grad的平均
        delta = -np.sqrt((Edelta + epsilon) / (Eg + epsilon)) * grad #用Eg計算
        x = x + delta
        Edelta = rho * Edelta + (1 - rho) * (delta**2)
        history.append(x)
    return history
def gradient_descent_RMSprop(df,x,alpha=0.01,beta = 0.9, iterations = 100,epsilon = 1e-8):
    history=[x]   
    v= np.ones_like(x) #初始值
    for i in range(iterations):
        if np.max(np.abs(df(x)))<epsilon:
            print("gradient足够小！")
            break
        grad = df(x)       
        v = beta*v+(1-beta)*grad**2       #計算gradient平方的運動平均 
        x = x-alpha*grad/(np.sqrt(v)+epsilon) #這裡，gradient被除以gradient平方運動平均的平方根，再乘以學習率。
        #這樣確保了在參數空間中每一步的移動不會太大，並允許學習率自適應地調整。
      
        history.append(x)
    return history
def gradient_descent_Adam(df, x, alpha=0.01, beta_1=0.9, beta_2=0.999, iterations=100, epsilon=1e-8):
    history = [x]
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, iterations + 1):  # 修改從1開始
        if np.max(np.abs(df(x))) < epsilon:
            print("gradient足够小！")
            break
        grad = df(x)
        m = beta_1 * m + (1 - beta_1) * grad  # 計算gradient的動量和平方值的運動平均
        v = beta_2 * v + (1 - beta_2) * grad**2  # 計算gradient的動量和平方值的運動平均
        
        m_1 = m / (1 - np.power(beta_1, t))
        v_1 = v / (1 - np.power(beta_2, t))  
        
        x = x - alpha * m_1 / (np.sqrt(v_1) + epsilon)
        
        history.append(x)
    return history
x0=np.array([3., 4.])
print("初始點",x0,"的gradient",df(x0))

path = gradient_descent(df,x0,0.000005,300000)
print("gradient_descent_basic",path[-1])

path_momentum = gradient_descent_momentum(df,x0,0.000005,0.8,300000)
print("gradient_descent_momentum",path_momentum[-1])

path_Adagrad = gradient_descent_Adagrad(df,x0,0.1,300000,1e-8)
print("gradient_descent_Adagrad",path_Adagrad[-1])

path_Adadelta = gradient_descent_Adadelta(df,x0,0.95,300000,1e-8)
print("gradient_descent_Adadelta",path_Adadelta[-1])

path_RMSprop = gradient_descent_RMSprop(df,x0,0.000005,0.99999999999,900000,1e-8)
print("gradient_descent_RMSprop",path_RMSprop[-1])

path_Adam = gradient_descent_Adam(df,x0,0.001,0.9,0.8,100000,1e-8)
print("gradient_descent_Adam",path_Adam[-1])

def plot_multiple_paths(paths, x, y, z, minima_, xmin, xmax, ymin, ymax):
    colors = ['k', 'b', 'g', 'r', 'm', 'y']  # 定義六種顏色
    labels = ['basic', 'momentum', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']  # 定義六種方法的名稱
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet) 

    # 使用迴圈畫出每一種方法的路徑
    for i, path in enumerate(paths):
        ax.quiver(path[:-1, 0], path[:-1, 1], path[1:, 0] - path[:-1, 0], path[1:, 1] - path[:-1, 1], 
                  scale_units='xy', angles='xy', scale=1, color=colors[i], label=labels[i])
    
    ax.plot(*minima_, 'r*', markersize=18)
    
    ax.legend()  # 顯示圖例
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')    
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    plt.show()

paths = [np.asarray(path), np.asarray(path_momentum), np.asarray(path_Adagrad), np.asarray(path_Adadelta), np.asarray(path_RMSprop), np.asarray(path_Adam)]
# 使用函數來畫出所有路徑
plot_multiple_paths(paths, x, y, z, minima_, xmin, xmax, ymin, ymax)