G = 60
Time =200
theta <- 0.9
eta <- 0.7
beta <- 0.95
delta <- 0
alpha <- 0.3
tol = 1/10^5
n <- 0.001
nsteps <- 100
k_history <- matrix(0, nrow = nsteps+1, ncol = 2*Time + 1)
# age-specified productivity
eps <- c(rep(1, G-20), rep(0, 20))


N <- matrix(0, nrow = 2* Time, ncol = 2* Time + G )
# population
N_0 <- 1

for(i in 1:nrow(N)){
  for(j in 1:ncol(N)){
    age <- G-j+i
    if(age >= 1 & age <= G){
      N[i,j] <- N_0 * (1+n)^(j-1)
    }
    
  }
}
# labor force
L <- rep(0, 2*Time)
for(t in 1:length(L)){
  
  L[t] <- sum(rev(eps)*N[t,t:(t+G-1)])
}


# steady state ----

get.steady.state <- function(K_initial = 1000, niter = 5000, eta = 0.8, tol = 1/10^2){
  K_steady <- K_initial
  K_new <- 0
  l <- 1
  while(l <= niter){#&((K_steady-K_new)/K_steady)>tol){
 
    k_steady <- K_steady/L[1]
    r_steady <- alpha*k_steady^(alpha-1) - delta
    w_steady <- (1-alpha)* k_steady^alpha 
    cons_steady <- rep(0, G)
    assets_steady <- rep(K_steady/G, G)
    
    
    
    
    
    wages <- eps[1]*w_steady
    
    
    for(i in 1:(G-1)){
      disc_factor <- 1
      for(j in 1:i){
        disc_factor <- disc_factor*(1+r_steady)
      }
      
      wages <- wages + eps[i+1]*w_steady/disc_factor
      
      
    }
    
    numenator <- wages
    
    
    denominator <- 1
    
    for(i in 1:(G-1)){
      
      
      disc_factor <- 1
      
      for(j in 1:i){
        
        disc_factor <- disc_factor*(1+r_steady)^(1/theta)/(1+r_steady)
      }
      
      
      
      disc_factor <- disc_factor*beta^(i/theta)
      
      
      denominator <- denominator + disc_factor
    }
    
    
    cons_steady[1] <- numenator/denominator
    
    for(i in 1:(G-1)){
      cons_steady[i+1] <- cons_steady[i]* (beta*(1+r_steady))^(1/theta)
    }
    assets_steady[1] <- eps[1]*w_steady - cons_steady[1]
    for(i in 2:(G)){
      assets_steady[i] <- (1+r_steady)*assets_steady[i-1] + eps[i]*w_steady - cons_steady[i]
      
    }
    
    K_new <- sum(assets_steady*rev(N[1,1:G]))
    K_steady <- eta * K_steady + (1-eta)* K_new
    
    
    l <- l+1
    
  }
  return(list(w = w_steady, r = r_steady, k = k_steady, K = K_steady, cons = cons_steady, assets= assets_steady))
}

ss <- get.steady.state(2000)


# factor prices
w <- c(rep(1, Time), rep(ss$w, Time))
r <- c(rep(1, Time), rep(ss$r, Time))
# capital
k_0 <- 8
k_steady <- ss$k
k <- c(seq(k_0, k_steady, length.out = Time/2 +1 ), rep(k_steady, 1.5*Time))
k_history[1,] <- k




Cons <- matrix(0, nrow = 2* Time, ncol = 2* Time + G )
A <- matrix(0, nrow = 2* Time, ncol = 2* Time + G )

# intitial endowment
A_initial <- rep(0, 2* Time+G)
A_initial[1:(G-1)] <- k_0*L[1]*(rev(ss$assets)[-1]/sum(rev(ss$assets)[-1]))




step <- 1
k_new <- 0


while(step <= nsteps){# & max((k - k_new)/ k) > tol){

  # начинаем с Time=1
  r <- alpha*k^(alpha-1) - delta
  w <- (1-alpha)* k^alpha 
  
  for(g in 1:(Time+G)){
    
    # дата рождения поколения
    bd <- max(g-G+1, 1)
    
    # годы жизни поколения внутри модели
    lt <- (bd:min(g, G+Time))
    
    # c_1,t
    if(length(lt) == 1){
      
      Cons[bd, g] <- A_initial[g]*(1+r[1])
      
      
    } else {
      # discounted wages
      wages <- eps[G-length(lt)+1]*w[lt[1]]
      
      
      for(j in lt[-1]){
        
        
        disc_factor <- 1
        
        for(l in lt[2]:j){
          
          
          disc_factor <- disc_factor*(1+r[l])
        }
        
        wages <- wages + eps[G+j-g]*w[j]/disc_factor
        
      }
      
      
      
      numenator <- A_initial[g]*(1+r[1]) + wages
      
      
      denominator <- 1
      
      for(j in lt[-1]){
        
        
        disc_factor <- 1
        
        for(l in lt[2]:j){
          
          disc_factor <- disc_factor*(1+r[l])^(1/theta)/(1+r[l])
        }
        
        
        
        disc_factor <- disc_factor*beta^((j-lt[1])/theta)
        
        
        
        denominator <- denominator + disc_factor
      }
      
      
      Cons[bd, g] <- numenator/denominator
      
      
      for(l in (lt[-1])){
        
        
        Cons[l, g] <- (beta*(1+r[l]))^(1/theta)*Cons[l-1, g]
        
      }
      
      
      
    }
    
    
    for(t in 1:(Time)){
      age <- G-g+t
      if(age > G|age < 1){
        A[t, g] <- 0
      } else {
        if(t == 1){
          A[t, g] <- A_initial[g]*(1+r[t]) - Cons[t, g] + eps[age]*w[t]
        }else{
          A[t, g] <- A[t-1, g]*(1+r[t]) - Cons[t, g] + eps[age]*w[t]
        }
        
        
        if(is.na(A[t, g])){
        }
        
      }
    }
    
  }
  
  
  k_new <- c(k_0, rowSums(A*N)/L)
  
  k_new[(Time+2):length(k_new)] <- k_steady

  
  k <- eta * k + (1-eta)  * k_new
  k_history[step+1,] <- k
  step <- step+1

}

df <- as.data.frame(k_history %>% t) %>%
  tibble::rownames_to_column('year') %>%
  melt %>% mutate(year = as.numeric(year)) 
ggplot()+
  geom_line(data = df %>% filter(year<=200), aes(x = year, y = value, group = variable))+
  geom_abline(slope = 0, intercept = ss$k, color = 'red')


plot(diag(Cons)[1:(Time)], type='l')
plot(Cons[1:60,60], type='l')
plot(A[2:61,61], type='l')
plot(r, type='l')
