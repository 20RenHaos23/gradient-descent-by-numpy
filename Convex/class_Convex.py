import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class Optimizator:
    def __init__(self,params):
        self.params = deepcopy(params)  # 使用深度複製
        #self.params = params
                
    def step(self,grads): 
       pass
    def parameters(self):
        return self.params
    
class Opt_Basic(Optimizator):
    def __init__(self,params,learning_rate):
        super().__init__(params)
        self.lr = learning_rate   
    def step(self,grads): 
        for i in range(len(self.params)):
            self.params[i] -= self.lr * grads[i]
        return self.params

class Opt_Momentum(Optimizator):
    def __init__(self,params,learning_rate,gamma):
        super().__init__(params)
        self.lr = learning_rate        
        self.gamma= gamma
        self.v = []
        for param in params:
            self.v.append(np.zeros_like(param) )
    def step(self,grads): 
        for i in range(len(self.params)):
            self.v[i] = self.gamma*self.v[i]+self.lr* grads[i]
            self.params[i] -= self.v[i]
        return self.params

class Opt_Adagrad(Optimizator):
    def __init__(self,params,learning_rate,epsilon):
        super().__init__(params)
        self.lr = learning_rate  
        self.epsilon = epsilon
        self.gl = []
        for param in params:
            self.gl.append(np.ones_like(param) * 0.1)
                
    def step(self,grads): 
        for i in range(len(self.params)):
            self.gl[i] += grads[i]**2
            self.params[i] -= self.lr * grads[i] / (np.sqrt(self.gl[i]) + self.epsilon)
        return self.params

class Opt_Adadelta(Optimizator):
    def __init__(self,params,rho,epsilon):
        super().__init__(params)
        self.rho = rho
        self.epsilon = epsilon
        self.Eg = []
        self.Edelta = []
        for param in params:
            self.Eg.append(np.zeros_like(param) )
            self.Edelta.append(np.zeros_like(param) )
                
    def step(self,grads): 
        for i in range(len(self.params)):
            self.Eg[i] = self.rho * self.Eg[i] + (1 - self.rho) * grads[i]**2
            delta = -np.sqrt((self.Edelta[i] + self.epsilon) / (self.Eg[i] + self.epsilon)) * grads[i]
            self.params[i] += delta
            self.Edelta[i] = self.rho * self.Edelta[i] + (1 - self.rho) * delta**2
        return self.params
    
class Opt_RMSprop(Optimizator):
    def __init__(self,params,learning_rate,beta,epsilon):
        super().__init__(params)
        self.lr = learning_rate        
        self.beta = beta
        self.epsilon = epsilon
        self.v = []
        for param in params:
            self.v.append(np.ones_like(param) )
                
    def step(self,grads): 
        
        for i in range(len(self.params)):
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * grads[i]**2
            self.params[i] -= self.lr * grads[i] / (np.sqrt(self.v[i]) + self.epsilon)
        return self.params

class Opt_Adam(Optimizator):
    def __init__(self,params,learning_rate,beta_1, beta_2, epsilon):
        super().__init__(params)
        self.lr = learning_rate        
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.timestep = 0  # 添加全局 timestep
        self.m = []
        self.v = []
        for param in params:
            self.m.append(np.zeros_like(param) )
            self.v.append(np.zeros_like(param) )
                
    def step(self,grads): 
        self.timestep += 1  # 在每次 step 時增加 timestep
        for i in range(len(self.params)):
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grads[i]
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * grads[i]**2

            m_1 = self.m[i] / (1 - np.power(self.beta_1, self.timestep))
            v_1 = self.v[i] / (1 - np.power(self.beta_2, self.timestep))

            self.params[i] -= self.lr * m_1 / (np.sqrt(v_1) + self.epsilon)
        return self.params


#-----------------------------------------------------------------------------------------------
#畫3D圖

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
def gradient_descent_(df,optimizator,iterations,epsilon = 1e-8):
    x, = optimizator.parameters()
    x = x.copy()
    history=[x]    
    for i in range(iterations):
        if np.max(np.abs(df(x)))<epsilon:
            print("gradient足够小！")
            break
        grad = df(x)
        x, = optimizator.step([grad])
        x = x.copy()
        history.append(x)
    return history


df = lambda x: np.array( [(1/8)*x[0],18*x[1]])
x0=np.array([-2.4, 0.2])
print("初始點",x0,"的gradient",df(x0))

optimizator_Basic = Opt_Basic([x0],0.1)
path_Basic = gradient_descent_(df,optimizator_Basic,100)
print("gradient_descent_basic",path_Basic[-1])

optimizator_Momentum = Opt_Momentum([x0],0.1,0.8)
path_Momentum = gradient_descent_(df,optimizator_Momentum,100)
print("gradient_descent_momentum",path_Momentum[-1])

optimizator_Adagrad = Opt_Adagrad([x0],0.5,1e-8)
path_Adagrad = gradient_descent_(df,optimizator_Adagrad,300)
print("gradient_descent_adagrad",path_Adagrad[-1])

optimizator_Adadelta = Opt_Adadelta([x0],0.9,1e-8,)
path_Adadelta = gradient_descent_(df,optimizator_Adadelta,100)
print("gradient_descent_adadelta",path_Adadelta[-1])

optimizator_RMSprop = Opt_RMSprop([x0],0.1,0.9,1e-8)
path_RMSprop = gradient_descent_(df,optimizator_RMSprop,100)
print("gradient_descent_rmsprop",path_RMSprop[-1])

optimizator_Adam = Opt_Adam([x0],0.1,0.7,0.9,1e-8)
path_Adam = gradient_descent_(df,optimizator_Adam,100)
print("gradient_descent_adam",path_Adam[-1])

paths = [np.asarray(path_Basic), np.asarray(path_Momentum), np.asarray(path_Adagrad), np.asarray(path_Adadelta), np.asarray(path_RMSprop), np.asarray(path_Adam)]


def plot_multiple_paths(paths, x, y, z, minima_, xmin, xmax, ymin, ymax):
    colors = ['k', 'b', 'g', 'r', 'm', 'y']  # 定義六種顏色
    labels = ['basic', 'momentum', 'Adagrad', 'Adadelta', 'RMSprop', 'Adam']  # 定義六種方法的名稱
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.contour(x, y, z, 50, cmap=plt.cm.jet)

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

paths = [np.asarray(path_Basic), np.asarray(path_Momentum), np.asarray(path_Adagrad), np.asarray(path_Adadelta), np.asarray(path_RMSprop), np.asarray(path_Adam)]
# 使用函數來畫出所有路徑
plot_multiple_paths(paths, x, y, z, minima_, xmin, xmax, ymin, ymax)