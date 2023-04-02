import matplotlib.pyplot as plt
import numpy as np
class Guess_plot:
    def __init__(self,model, t_0, t_1):
        self.model=model
        self.time = range(t_0,t_1)
    def plot(self):
        self.ax.plot()
        
class Aggregate_plot(Guess_plot):
    def __init__(self,model,t_0=2, t_1=100, name = ""):
        super(Aggregate_plot, self).__init__(model, t_0, t_1)
        self.fig, self.ax = plt.subplots(3,3, figsize = (15,10))
        self.fig.suptitle(name, fontsize=10)
    def create(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle, "color":color}
        c = self.model.Consumption[self.time]/(self.model.N_epsilon[self.time]*self.model.A[0,self.time])
        l = self.model.Labor[self.time]/self.model.N_epsilon[self.time]

        self.ax[0,0].plot(self.model.k[0,self.time], **plt_kwargs, label = r"$k_N$")
        self.ax[1,0].plot(self.model.k[1,self.time], **plt_kwargs, label = r"$k_E$")
        self.ax[0,1].plot(self.model.i[0,self.time], **plt_kwargs, label = r"$i_N$")
        self.ax[1,1].plot(self.model.i[1,self.time], **plt_kwargs, label = r"$i_E$")
        self.ax[0,2].plot(c, **plt_kwargs, label = r"$c$")
        self.ax[1,2].plot(l, **plt_kwargs, label = r"$l$")
        self.ax[2,0].plot(self.model.w[self.time]/self.model.A[0,self.time], **plt_kwargs, label = r"$\hat{w}$")
        self.ax[2,1].plot(self.model.price[self.time],**plt_kwargs, label = r"$p$")
        self.ax[2,2].plot(self.model.l_demand[0, self.time], **plt_kwargs, label = r"$l_0$")
        for row in self.ax:
            for col in row:
                col.legend()
                
                
    def update(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle, "color":color}
        c = self.model.Consumption[self.time]/(self.model.N_epsilon[self.time]*self.model.A[0,self.time])
        l = self.model.Labor[self.time]/self.model.N_epsilon[self.time]
        
        self.ax[0,0].plot(self.model.k[0,self.time], **plt_kwargs)
        self.ax[1,0].plot(self.model.k[1,self.time], **plt_kwargs)
        self.ax[0,1].plot(self.model.i[0,self.time], **plt_kwargs)
        self.ax[1,1].plot(self.model.i[1,self.time], **plt_kwargs)
        self.ax[0,2].plot(c, **plt_kwargs)
        self.ax[1,2].plot(l, **plt_kwargs)
        self.ax[2,0].plot(self.model.w[self.time]/self.model.A[0,self.time], **plt_kwargs)
        self.ax[2,1].plot(self.model.price[self.time],**plt_kwargs)
        self.ax[2,2].plot(self.model.l_demand[0, self.time], **plt_kwargs)
        for row in self.ax:
            for col in row:
                col.legend()

class Household_plot(Guess_plot):
    def __init__(self,model,g_0=30, g_1=60, name = ""):
        super(Household_plot, self).__init__(model, 0, 0)
        self.fig, self.ax = plt.subplots(2,3, figsize = (15,10))
        self.g_0 = g_0
        self.g_1 = g_1
        self.fig.suptitle(name, fontsize=10)
    def create(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle}
        
        time_0 = range(max(1, self.g_0-self.model.G+1), self.g_0)
        time_1 = range(max(1, self.g_1-self.model.G+1), self.g_1)
        self.ax[0,0].plot(self.model.c[0,self.g_0,time_0], **plt_kwargs,color="red", label = r"$c_{f,0}$")
        self.ax[0,1].plot(self.model.l[0,self.g_0,time_0], **plt_kwargs,color="red", label = r"$l_{f,0}$")
        self.ax[0,2].plot(self.model.a[0,self.g_0,range(max(1, self.g_0-self.model.G+1), self.g_0+1)], **plt_kwargs,color="red", label = r"$a_{f,0}$")
        self.ax[0,0].plot(self.model.c[1,self.g_0,time_0], **plt_kwargs,color="black", label = r"$c_{m,0}$")
        self.ax[0,1].plot(self.model.l[1,self.g_0,time_0], **plt_kwargs,color="black", label = r"$l_{m,0}$")
        self.ax[0,2].plot(self.model.a[1,self.g_0,range(max(1, self.g_0-self.model.G+1), self.g_0+1)], **plt_kwargs,color="black", label = r"$a_{m,0}$")
        
        self.ax[1,0].plot(self.model.c[0,self.g_1,time_1], **plt_kwargs,color="red", label = r"$c_{f,1}$")
        self.ax[1,1].plot(self.model.l[0,self.g_1,time_1], **plt_kwargs,color="red", label = r"$l_{f,1}$")
        self.ax[1,2].plot(self.model.a[0,self.g_1,range(max(1, self.g_1-self.model.G+1), self.g_1+1)], **plt_kwargs,color="red", label = r"$a_{f,1}$")
        self.ax[1,0].plot(self.model.c[1,self.g_1,time_1], **plt_kwargs,color="black", label = r"$c_{m,1}$")
        self.ax[1,1].plot(self.model.l[1,self.g_1,time_1], **plt_kwargs,color="black", label = r"$l_{m,1}$")
        self.ax[1,2].plot(self.model.a[1,self.g_1,range(max(1, self.g_1-self.model.G+1), self.g_1+1)], **plt_kwargs,color="black", label = r"$a_{m,1}$")
            
        for row in self.ax:
            for col in row:
                col.legend()

    def update(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle}
        
        time_0 = range(max(1, self.g_0-self.model.G+1), self.g_0)
        time_1 = range(max(1, self.g_1-self.model.G+1), self.g_1)
        self.ax[0,0].plot(self.model.c[0,self.g_0,time_0], **plt_kwargs,color="red")
        self.ax[0,1].plot(self.model.l[0,self.g_0,time_0], **plt_kwargs,color="red")
        self.ax[0,2].plot(self.model.a[0,self.g_0,range(max(1, self.g_0-self.model.G+1), self.g_0+1)], **plt_kwargs,color="red")
        self.ax[0,0].plot(self.model.c[1,self.g_0,time_0], **plt_kwargs,color="black")
        self.ax[0,1].plot(self.model.l[1,self.g_0,time_0], **plt_kwargs,color="black")
        self.ax[0,2].plot(self.model.a[1,self.g_0,range(max(1, self.g_0-self.model.G+1), self.g_0+1)], **plt_kwargs,color="black")
        
        self.ax[1,0].plot(self.model.c[0,self.g_1,time_1], **plt_kwargs,color="red")
        self.ax[1,1].plot(self.model.l[0,self.g_1,time_1], **plt_kwargs,color="red")
        self.ax[1,2].plot(self.model.a[0,self.g_1,range(max(1, self.g_1-self.model.G+1), self.g_1+1)], **plt_kwargs,color="red")
        self.ax[1,0].plot(self.model.c[1,self.g_1,time_1], **plt_kwargs,color="black")
        self.ax[1,1].plot(self.model.l[1,self.g_1,time_1], **plt_kwargs,color="black")
        self.ax[1,2].plot(self.model.a[1,self.g_1,range(max(1, self.g_1-self.model.G+1), self.g_1+1)], **plt_kwargs,color="black")

        
        
        
class Gov_plot(Guess_plot):
    def __init__(self,model,t_0=2, t_1=100, name = ""):
        super(Gov_plot, self).__init__(model, t_0, t_1)
        self.fig, self.ax = plt.subplots(2,2, figsize = (15,10))
        self.fig.suptitle(name, fontsize=10)
        self.labels = [[[r"$\hat{Gov}$", r"$\hat{Debt}$"], [r"$\tau_{I}$",r"$\tau_{VA}$"]],
                       [[r"$\hat{Deficit_\rho}$"],[r"$\hat{Deficit}$"]]]
        self.colors = [[["black", "blue"], ["black","blue"]],
                       [["black"],["black"]]]
        
        
    @property
    def series(self):
        return [[[self.model.Gov_to_GDP[self.time],self.model.Debt_to_GDP[self.time]]
                 , [self.model.tau_I[self.time], self.model.tau_VA[self.time]]
                ],
                 [[self.model.Deficit_rho_to_GDP[self.time]]
                  ,[self.model.Deficit_to_GDP[self.time]]
                 ]
                ]
    def create(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle}
        
        for labels_row,series_row,color_row, ax_row in zip(self.labels,self.series,self.colors,  self.ax):
            for labels_col, series_col, color_col, ax_col  in zip(labels_row, series_row,color_row, ax_row):
                for l, s, c in zip(labels_col, series_col, color_col):
                    ax_col.plot(s, color = c, **plt_kwargs,label = l)
                ax_col.legend()
        # self.ax[0,0].set_ylim([0., 1.])    
                
    def update(self,alpha=1, linestyle='solid', color="black"):
        plt_kwargs = {"alpha":alpha, "linestyle":linestyle}
        
        for labels_row,series_row,color_row, ax_row in zip(self.labels,self.series,self.colors,  self.ax):
            for labels_col, series_col, color_col, ax_col  in zip(labels_row, series_row,color_row, ax_row):
                for l, s, c in zip(labels_col, series_col, color_col):
                    ax_col.plot(s, color = c, **plt_kwargs)
                    

                    
                    

