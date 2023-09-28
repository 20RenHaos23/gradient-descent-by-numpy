'''
Adadelta 是另一種適應性學習率的優化方法。它是為了克服 Adagrad 方法中學習率過快衰減的問題而被提出的
Adadelta 的一個重要特點是它不需要設置外部學習率，因為它已經基於gradient的運動平均來調整學習率。

Adagrad 的主要問題是其學習率會隨著時間持續遞減，因為它累積了所有過去gradient的平方。
在長時間的訓練過程中，學習率可能會過小，導致模型的參數更新非常緩慢，甚至停滯不前。
'''
import numpy as np
import matplotlib.pyplot as plt



f = lambda x, y: (1/16)*x**2+9*y**2

#定義函數的最小值點:
minima = np.array([0., 0.])
minima_ = minima.reshape(-1, 1)
#建立格點座標:
xmin, xmax = -2.5, 2.5
ymin, ymax = -0.5, 0.5
N = 1000

xstep = (xmax - xmin) / (N-1)
ystep = (ymax - ymin) / (N-1)

x_list = np.arange(xmin, xmax + xstep/2, xstep)  # 這裡我們加了xstep的一半來確保xmax包含在裡面
y_list = np.arange(ymin, ymax + ystep/2, ystep)  # 同上，確保ymax包含在裡面
x, y = np.meshgrid(x_list, y_list)
#計算函數值:
z = f(x, y)
#繪製三維曲面圖:
fig = plt.figure(figsize=(8, 5))
ax = plt.axes(projection='3d', elev=50, azim=-50)
ax.plot_surface(x, y, z, alpha=.8, cmap=plt.cm.jet)
#ax.plot_surface(x, y, z, norm=LogNorm(), rstride=1, cstride=1, 
#               edgecolor='none', alpha=.8, cmap=plt.cm.jet)
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
#-----------------------------------------------------------------------------------------------

#畫等高線圖
# 計算方程式的偏導數
df_x = lambda x, y: (1/8)*x
df_y = lambda x, y: 18*y

# 計算方程式在格點座標上的偏導數
dz_dx = df_x(x, y)
dz_dy = df_y(x, y)
#創建畫布和軸:
fig, ax = plt.subplots(figsize=(10, 6))

#繪製等高線圖:

ax.contour(x, y, z, 50, cmap=plt.cm.jet)
#levels 參數用於定義等高線的高度
#np.logspace(0, 5, 35)生成35個在 [10**0 (也就是1), 10**5] 範圍內的數字
#norm=LogNorm() 使得等高線在對數刻度上均勻分布。

#繪製gradient向量:
#ax.quiver(x, y, x - dz_dx, y - dz_dy, alpha=.5) #不畫 會整個都是箭頭的顏色

#繪製最小值點:
ax.plot(*minima_, 'r*', markersize=18)
#設定軸標籤和範圍:
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))
#顯示圖形:
plt.show()
#-----------------------------------------------------------------------------------------------


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

df = lambda x: np.array( [(1/8)*x[0],18*x[1]])
x0=np.array([-2.4, 0.2])
print("初始點",x0,"的gradient",df(x0))

path = gradient_descent_Adadelta(df,x0,0.01,4000,1e-8)
print("找到gradient最小的座標位置",path[-1])

def plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(x, y, z, 50, cmap=plt.cm.jet) #再畫一次圖
    
    ax.quiver(path[:-1,0], path[:-1,1], path[1:,0]-path[:-1,0], path[1:,1]-path[:-1,1], scale_units='xy', angles='xy', scale=1, color='k')
    ax.plot(*minima_, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')    
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

path = np.asarray(path) 
plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax)