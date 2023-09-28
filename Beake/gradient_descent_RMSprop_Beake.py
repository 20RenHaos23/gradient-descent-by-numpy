'''
RMSprop (Root Mean Square Propagation) 主要被引入來解決 Adagrad 方法在訓練深度神經網絡時學習率快速衰減的問題。
RMSprop (Root Mean Square Propagation) 是一種適應性學習率的優化方法，它主要被引入來解決 Adagrad 方法在訓練深度神經網絡時學習率快速衰減的問題。
'''
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

x0=np.array([3., 4.])
print("初始點",x0,"的gradient",df(x0))

path = gradient_descent_RMSprop(df,x0,0.000005,0.99999999999,900000,1e-8)
print("找到gradient最小的座標位置",path[-1])

def plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.jet) #再畫一次圖
    
    ax.quiver(path[:-1,0], path[:-1,1], path[1:,0]-path[:-1,0], path[1:,1]-path[:-1,1], scale_units='xy', angles='xy', scale=1, color='k')
    ax.plot(*minima_, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')    
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

path = np.asarray(path) 
plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax)