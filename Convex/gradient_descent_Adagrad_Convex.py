'''
Adagrad 的主要特點是其學習率會隨著時間遞減。
對於出現頻率高的特徵，其學習率會更快地衰減，
而對於出現頻率低的特徵，學習率則相對較慢地衰減。
這使得 Adagrad 特別適用於稀疏數據。不過，由於gradient的平方和持續累積，
這可能會導致學習率在訓練後期過於衰減，進而使得訓練過程停滯。
這也是後續方法如 Adadelta 和 RMSprop 被提出的原因，它們旨在解決這一問題。
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

df = lambda x: np.array( [(1/8)*x[0],18*x[1]])
x0=np.array([-2.4, 0.2])
print("初始點",x0,"的gradient",df(x0))

path = gradient_descent_Adagrad(df,x0,0.5,300,1e-8)
print("找到gradient最小的座標位置",path[-1])

def plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(x, y, z, 50, cmap=plt.cm.jet)
    
    ax.quiver(path[:-1,0], path[:-1,1], path[1:,0]-path[:-1,0], path[1:,1]-path[:-1,1], scale_units='xy', angles='xy', scale=1, color='k')
    ax.plot(*minima_, 'r*', markersize=18)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')    
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

path = np.asarray(path) 
plot_path(path,x,y,z,minima_,xmin, xmax,ymin, ymax)