import matplotlib.pyplot as plt
import numpy as np
class Guess_plot:
    def __init__(self,olg, t_0, t_1):
        self.olg=olg
        self.time = range(t_0,t_1)
    def plot(self):
        self.ax.plot()
        
class Aggregate_plot(Guess_plot):
    def __init__(self,olg,t_0=2, t_1=100):
        super(Aggregate_plot, self).__init__(olg, t_0, t_1)
        self.fig, self.ax = plt.subplots(3,3, figsize = (15,10))
    def create(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle, "color":color}
        c = self.olg.Consumption[self.time]/(self.olg.N[:,:,self.time].sum(axis=(0,1))*self.olg.A[0,self.time])
        l = self.olg.Labor[self.time]/self.olg.N[:,:,self.time].sum()
        
        self.ax[0,0].plot(self.olg.k[0,self.time], **plt_kwargs, label = r"$k_N$")
        self.ax[1,0].plot(self.olg.k[1,self.time], **plt_kwargs, label = r"$k_E$")
        self.ax[0,1].plot(self.olg.i[0,self.time], **plt_kwargs, label = r"$i_N$")
        self.ax[1,1].plot(self.olg.i[1,self.time], **plt_kwargs, label = r"$i_E$")
        self.ax[0,2].plot(l, **plt_kwargs, label = r"$c$")
        self.ax[1,2].plot(c, **plt_kwargs, label = r"$l$")
        self.ax[2,0].plot(self.olg.w[self.time], **plt_kwargs, label = r"$w$")
        self.ax[2,1].plot(self.olg.price[self.time],**plt_kwargs, label = r"$p$")
        self.ax[2,2].plot(self.olg.price[self.time],**plt_kwargs, label = r"$p_N$")
        for row in self.ax:
            for col in row:
                col.legend()
                
                
    def update(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle, "color":color}
        c = self.olg.Consumption[self.time]/(self.olg.N[:,:,self.time].sum(axis=(0,1))*self.olg.A[0,self.time])
        l = self.olg.Labor[self.time]/self.olg.N[:,:,self.time].sum()
        
        self.ax[0,0].plot(self.olg.k[0,self.time], **plt_kwargs)
        self.ax[1,0].plot(self.olg.k[1,self.time], **plt_kwargs)
        self.ax[0,1].plot(self.olg.i[0,self.time], **plt_kwargs)
        self.ax[1,1].plot(self.olg.i[1,self.time], **plt_kwargs)
        self.ax[0,2].plot(l, **plt_kwargs)
        self.ax[1,2].plot(c, **plt_kwargs)
        self.ax[2,0].plot(self.olg.w[self.time], **plt_kwargs)
        self.ax[2,1].plot(self.olg.price[self.time],**plt_kwargs)
        self.ax[2,2].plot(self.olg.price[self.time],**plt_kwargs)
        for row in self.ax:
            for col in row:
                col.legend()

class Household_plot(Guess_plot):
    def __init__(self,olg,g_0=30, g_1=60):
        super(Household_plot, self).__init__(olg, 0, 0)
        self.fig, self.ax = plt.subplots(2,3, figsize = (15,10))
        self.g_0 = g_0
        self.g_1 = g_1
    def create(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle}
        
        time_0 = range(max(1, self.g_0-self.olg.G+1), self.g_0)
        time_1 = range(max(1, self.g_1-self.olg.G+1), self.g_1)
        self.ax[0,0].plot(self.olg.c[0,self.g_0,time_0], **plt_kwargs,color="red", label = r"$c_{f,0}$")
        self.ax[0,1].plot(self.olg.l[0,self.g_0,time_0], **plt_kwargs,color="red", label = r"$l_{f,0}$")
        self.ax[0,2].plot(self.olg.a[0,self.g_0,range(max(1, self.g_0-self.olg.G+1), self.g_0+1)], **plt_kwargs,color="red", label = r"$a_{f,0}$")
        self.ax[0,0].plot(self.olg.c[1,self.g_0,time_0], **plt_kwargs,color="black", label = r"$c_{m,0}$")
        self.ax[0,1].plot(self.olg.l[1,self.g_0,time_0], **plt_kwargs,color="black", label = r"$l_{m,0}$")
        self.ax[0,2].plot(self.olg.a[1,self.g_0,range(max(1, self.g_0-self.olg.G+1), self.g_0+1)], **plt_kwargs,color="black", label = r"$a_{m,0}$")
        
        self.ax[1,0].plot(self.olg.c[0,self.g_1,time_1], **plt_kwargs,color="red", label = r"$c_{f,1}$")
        self.ax[1,1].plot(self.olg.l[0,self.g_1,time_1], **plt_kwargs,color="red", label = r"$l_{f,1}$")
        self.ax[1,2].plot(self.olg.a[0,self.g_1,range(max(1, self.g_1-self.olg.G+1), self.g_1+1)], **plt_kwargs,color="red", label = r"$a_{f,1}$")
        self.ax[1,0].plot(self.olg.c[1,self.g_1,time_1], **plt_kwargs,color="black", label = r"$c_{m,1}$")
        self.ax[1,1].plot(self.olg.l[1,self.g_1,time_1], **plt_kwargs,color="black", label = r"$l_{m,1}$")
        self.ax[1,2].plot(self.olg.a[1,self.g_1,range(max(1, self.g_1-self.olg.G+1), self.g_1+1)], **plt_kwargs,color="black", label = r"$a_{m,1}$")
            
        for row in self.ax:
            for col in row:
                col.legend()

    def update(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle}
        
        time_0 = range(max(1, self.g_0-self.olg.G+1), self.g_0)
        time_1 = range(max(1, self.g_1-self.olg.G+1), self.g_1)
        self.ax[0,0].plot(self.olg.c[0,self.g_0,time_0], **plt_kwargs,color="red")
        self.ax[0,1].plot(self.olg.l[0,self.g_0,time_0], **plt_kwargs,color="red")
        self.ax[0,2].plot(self.olg.a[0,self.g_0,range(max(1, self.g_0-self.olg.G+1), self.g_0+1)], **plt_kwargs,color="red")
        self.ax[0,0].plot(self.olg.c[1,self.g_0,time_0], **plt_kwargs,color="black")
        self.ax[0,1].plot(self.olg.l[1,self.g_0,time_0], **plt_kwargs,color="black")
        self.ax[0,2].plot(self.olg.a[1,self.g_0,range(max(1, self.g_0-self.olg.G+1), self.g_0+1)], **plt_kwargs,color="black")
        
        self.ax[1,0].plot(self.olg.c[0,self.g_1,time_1], **plt_kwargs,color="red")
        self.ax[1,1].plot(self.olg.l[0,self.g_1,time_1], **plt_kwargs,color="red")
        self.ax[1,2].plot(self.olg.a[0,self.g_1,range(max(1, self.g_1-self.olg.G+1), self.g_1+1)], **plt_kwargs,color="red")
        self.ax[1,0].plot(self.olg.c[1,self.g_1,time_1], **plt_kwargs,color="black")
        self.ax[1,1].plot(self.olg.l[1,self.g_1,time_1], **plt_kwargs,color="black")
        self.ax[1,2].plot(self.olg.a[1,self.g_1,range(max(1, self.g_1-self.olg.G+1), self.g_1+1)], **plt_kwargs,color="black")

        
        
        
class Gov_plot(Guess_plot):
    def __init__(self,olg,t_0=2, t_1=100):
        super(Gov_plot, self).__init__(olg, t_0, t_1)
        self.fig, self.ax = plt.subplots(2,2, figsize = (15,10))
    def create(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle, "color":color}
        rho_deficit_ratio = (self.olg.Rho_sum - self.olg.sigma*self.olg.w * np.sum(self.olg.rho[:,:,:]*self.olg.N[:,:,:], axis = (0,1)))/self.olg.GDP
        self.ax[0,0].plot(self.olg.Gov[self.time], **plt_kwargs, label = r"$Gov$")
        self.ax[0,1].plot(self.olg.sigma[self.time], **plt_kwargs, label = r"$\sigma$")
        self.ax[1,0].plot(rho_deficit_ratio[self.time], **plt_kwargs, label = r"$\rho$ Deficit to GDP")
        self.ax[1,1].plot(self.olg.Deficit_ratio[self.time], **plt_kwargs, label = r"Deficit to GDP")
        
        
        for row in self.ax:
            for col in row:
                col.legend()
                
                
    def update(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle, "color":color}
        
        rho_deficit_ratio = (self.olg.Rho_sum - self.olg.sigma*self.olg.w *\
                             np.sum(self.olg.rho[:,:,:]*self.olg.N[:,:,:], axis = (0,1)))/self.olg.GDP
        
        self.ax[0,0].plot(self.olg.Gov[self.time], **plt_kwargs)
        self.ax[0,1].plot(self.olg.sigma[self.time], **plt_kwargs)
        self.ax[1,0].plot(rho_deficit_ratio[self.time], **plt_kwargs)
        self.ax[1,1].plot(self.olg.Deficit_ratio[self.time], **plt_kwargs)