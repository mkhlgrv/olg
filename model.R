source('lib.R')
# Two-period OLG ——
n =0.05
beta = 0.3
alpha = 0.2
delta = 0.99
t <- 20

k <- Y_outcome <- Y_income <- Y_prod <- K<- L <- c_young <- c_old <- w <- s <-r <- rep(NA, t)
L[1] <-1
K[1] <- 0.1
k[1] <- K[1]/L[1]

k_star <- ((1-alpha)*beta/((1+beta)*(1+n) - (1-delta)))^(1/(1-alpha))

for(i in 2:t){
  k[i] <- (beta *(1-alpha)*k[i-1]^alpha + (1-delta) * k[i-1])/((1+beta)*(1+n))
  w[i] <- k[i]^alpha * (1-alpha) - (1-delta) * k[i]
  r[i] <- alpha * k[i] ^ (alpha-1) +1 - delta
  c_young[i] <- w[i]/(1+beta)
  s[i] <- w[i]*beta /(1+beta)
  if(i > 2){
    c_old[i] <- s[i-1]*r[i]
  }
  L[i] <-L[i-1]*(1+n)
  K[i] <- k[i]*L[i]
  
  
  Y_income[i] <- r[i] * K[i] + w[i] * L[i]
  Y_prod[i] <- K[i] ^ (alpha) * L[i] ^ (1-alpha)
  if(i > 2){
    Y_outcome[i] <- (c_young[i] + s[i]) * L[i] + c_old[i] * L[i-1]
  }
  
}

ggplot(mapping = aes(x = 1:t))+
  geom_line(aes(y = k, color = 'k'))+
  # geom_line(aes(y = c_young, color = 'c_young'))+
  # geom_line(aes(y = c_old, color = 'c_old'))+
  # geom_line(aes(y = s, color = 's'))+
  geom_line(aes(y = k_star), linetype = 2)


ggplot()+
  geom_point(aes(x = k, y = lead(k), color = 'k'))

ggplot(mapping = aes(x = 1:t))+
  geom_line(aes(y = K, color = 'K'))+
  geom_line(aes(y = L, color = 'L'))

Cons = (c_young * L) + (c_old * lag(L))
Inv <- s * L
Y_out <- Cons + Inv
Y_prod <- K^alpha * L^(1-alpha)

# Three-Period ——
# k_{t+1} = s_t^{young}/(1+n+d) + s_t^{old}/((1+n)(1+n+d)) + k_t * (1-delta)/(1+n)
# где s_t^{young} = w_t * (1+beta)*beta - (w_{t+1}/ R_{t+1}) * d/ (1+beta+beta^2)
# где s_t^{old} = beta /(1+beta)* (d* w_t + (w_{t-1} * R_t)) *beta * (1+beta)/(1+beta+beta^2) - w_t * d/(1+beta+beta^2))
# w_t = k_t^{alpha} - k_t * (1-delta + alpha * k_t^{alpha - 1})
# R_t = (1-delta + alpha * k_t^{alpha - 1})

# R_t * w_{t-1} = (1-delta + alpha * k_t^{alpha - 1})*
# k_{t-1}^{alpha} (1-alpha) - k_{t-1} * (1-delta)


# коэффициенты (1-ый уровень)
# при w_t: (beta + beta^2 + beta^2 * d / (1+n))/((1+n+d)*(1+beta+beta^2))
# при R_t * w_{t-1}: beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))
# при w_{t+1}/ R_{t+1}: -d/ ((1+n+d)*(1+beta+beta^2))
# при k_t (амортизация, только часть коэффициента от всего уравнения): (1-delta)/ (1+n)
# коэффициенты (2-ой уровень)
#
# из w_t (плюс добавил в коэффициент при k_t сразу часть из амортизации)
# k[t]^alpha * (1-alpha) * (beta + beta^2 + beta^2 * d / (1+n))/((1+n+d)*(1+beta+beta^2))
# k[t] * ((delta-1) * (beta + beta^2 + beta^2 * d / (1+n))/((1+n+d)*(1+beta+beta^2)) +
# (1-delta)/(1+n))
#
# из R_t * w_{t-1}
# k[t-1]^alpha * (1-delta) * (1-alpha) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))
# k[t-1] * (-1) * (1-delta)^2 * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))
# k[t]^(alpha-1) * k[t-1]^alpha * alpha * (1-alpha) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))
# k[t]^(alpha-1) * k[t-1] * alpha * (delta-1) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))
#
# из w_{t+1}/ R_{t+1}
# k[t+1] * d / ((1+n+d)*(1+beta+beta^2))
# k[t+1]^alpha / ( 1- delta + alpha * k[t+1]^(alpha - 1)) * (-d)/ ((1+n+d)*(1+beta+beta^2))



# тогда общее уравнение динамики
# 0 = - k_[t+1]+ k[t]^alpha * (1-alpha) * (beta + beta^2 + beta^2 * d / (1+n))/((1+n+d)*(1+beta+beta^2))+
# k[t] * ((delta-1) * (beta + beta^2 + beta^2 * d / (1+n))/((1+n+d)*(1+beta+beta^2)) + (1-delta)/(1+n))+
# k[t-1]^alpha * (1-delta) * (1-alpha) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))+
# k[t-1] * (-1) * (1-delta)^2 * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))+
# k[t]^(alpha-1) * k[t-1]^alpha * alpha * (1-alpha) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))+
# k[t]^(alpha-1) * k[t-1] * alpha * (delta-1) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))+
# k[t+1] * d / ((1+n+d)*(1+beta+beta^2))+
# k[t+1]^alpha / ( 1- delta + alpha * k[t+1]^(alpha - 1)) * (-d)/ ((1+n+d)*(1+beta+beta^2))


capital <- function(k, alpha = 0.5, beta = 0.8, d = 1, n = 0.01, delta = 0.9){
  -(- k+ k^alpha * (1-alpha) * (beta + beta^2 + beta^2 * d / (1+n))/((1+n+d)*(1+beta+beta^2))+
      k * ((delta-1) * (beta + beta^2 + beta^2 * d / (1+n))/((1+n+d)*(1+beta+beta^2)) + (1-delta)/(1+n))+
      k^alpha * (1-delta) * (1-alpha) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))+
      k * (-1) * (1-delta)^2 * beta^2 /
      ((1+n)*(1+n+d)*(1+beta+beta^2))+
      k^(alpha-1) * k^alpha * alpha * (1-alpha) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))+
      k^(alpha-1) * k * alpha * (delta-1) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))+
      k * d / ((1+n+d)*(1+beta+beta^2))+
      k^alpha / ( 1- delta + alpha * k^(alpha - 1)) * (-d)/ ((1+n+d)*(1+beta+beta^2))
  )
}
# f(k*) = 0
curve(delta(x), 0,.5)
abline(h = 0, lty = 3)
# находим корни
capital_ss <- uniroot(capital, c(1/10^100, 1))$root
capital_ss
# моделируем переход к steady state
# 1 шаг от начального капитала к устойчивому состоянию
# задать Time = 50, k[0]
# найти k[t], L[t]
# найти r[t], w[t]
# найти s[t]
# k[t]

Time <- 50
alpha = 0.5
beta = 0.8
d = 1
n = 0.01
delta = 0.9

# labour force ——
Capital <- capital <- L <- N <- rep(NA, Time)
N[1] <- 1
for(t in 2:Time){
  N[t] <- N[1] * (1+n)^(t-1)
}

for(t in 1:Time){
  L[t] <- N[1] * (1+n+d) * (1+n)^(t-2)
}

# capital ——
capital[1] <- 0.1


# linear guess

for(t in 2:Time){
  capital[t] <- capital[t-1] + (capital_ss-capital[1])/(Time-1)
}
Capital <- L*capital

ggplot(mapping = aes(x = 1:Time))+
  geom_line(aes(y = Capital, color = 'K'))+
  geom_line(aes(y = L, color = 'L'))





ggplot(mapping = aes(x = 1:Time))+
  
  geom_line(aes(y = capital, color = 'capital'))


update.path <- function(capital,
                        step = 1,
                        eta = 0.9,
                        alpha = 0.5,
                        beta = 0.8,
                        d = 1,
                        n = 0.01,
                        capital0 = 0.1,
                        delta = 0.9,
                        Time = 200
){
  
  capital_lag <- c(capital[1] * 1,capital[1] * 1)
  capital_old <- capital
  
  # R, w ——
  
  R <- w <- rep(NA, Time)
  for(t in 1:Time){
    R[t] <- 1 - delta + alpha * capital[t]^(alpha-1)
    w[t] <- capital[t]^alpha - capital[t]*R[t]
  }
  
  # savings, consumption ——
  c_young <- c_medium <- c_old <- s_young <- s_medium <- rep(NA, Time)
  
  for(t in 2:Time){
    s_young[t] <- w[t] * (1+beta)*beta/ (1+beta+beta^2) - (w[t+1]/ R[t+1]) * d/ (1+beta+beta^2)
    s_medium[t] <- beta /(1+beta)*
      (d* w[t] + w[t-1] * R[t] *beta * (1+beta)/(1+beta+beta^2) - w[t] * d/(1+beta+beta^2))
  }
  # проблема первой точки
  
  # нельзя посчитать выбор старшего и среднего поколения без 2 значений предыдущего капитала
  # capital_lag[i] = k_{-i} то есть в обратном порядке
  
  R_lag <- 1 - delta + alpha * capital_lag^(alpha-1)
  w_lag <- capital_lag^alpha - capital_lag*R_lag
  
  s_young[1] <-
    w[1] * (1+beta)*beta/ (1+beta+beta^2) - (w[2]/ R[2]) * d/ (1+beta+beta^2)
  
  s_young0 <- w_lag[1] *
    (1+beta)*beta/ (1+beta+beta^2) - (w[1]/ R[1]) * d/ (1+beta+beta^2)
  
  s_medium[1] <- beta /(1+beta)*
    (d* w[1] + w_lag[1] * R[1] *beta *
       (1+beta)/(1+beta+beta^2) - w[1] * d/(1+beta+beta^2))
  
  s_medium0 <- beta /(1+beta)*
    (d* w_lag[1] + w_lag[2] * R_lag[1] *beta *
       (1+beta)/(1+beta+beta^2) - w_lag[1] * d/(1+beta+beta^2))
  
  
  # последняя точка
  s_young[Time] <- w[Time] * (1+beta)*beta/ (1+beta+beta^2) - (w[Time]/ R[Time]) * d/ (1+beta+beta^2)
  
  
  
  c_young <- w-s_young
  c_medium <- d*w + R*c(s_young0, s_young[1:(Time-1)]) - s_medium
  c_old <- R*c(s_medium0, s_medium[1:(Time-1)])
  
  
  
  # new path ——
  
  for(t in 2:Time){
    capital[t] = s_young[t-1] /(1+n+d)+
      s_medium[t-1]/((1+n)*(1+n+d)) +
      capital[t-1]* (1-delta)/(1+n)
    
  }
  
  return(
    data.frame(step,
               alpha,
               beta,
               d,
               n,
               delta,
               capital0,
               t = 1:Time,
               s_young, s_medium, c_young,
               c_medium, c_old, w, R,
               capital =
                 (1-eta)* capital + eta*capital_old
    )
  )
  
}



capital.eq <- function(k, alpha = 0.5, beta = 0.8, d = 1, n = 0.01, delta = 0.9){
  -(- k+ k^alpha * (1-alpha) * (beta + beta^2 + beta^2 * d / (1+n))/((1+n+d)*(1+beta+beta^2))+
      k * ((delta-1) * (beta + beta^2 + beta^2 * d / (1+n))/((1+n+d)*(1+beta+beta^2)) +
             (1-delta)/(1+n))+
      k^alpha * (1-delta) * (1-alpha) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))+
      k * (-1) * (1-delta)^2 * beta^2 /
      ((1+n)*(1+n+d)*(1+beta+beta^2))+
      k^(alpha-1) * k^alpha * alpha * (1-alpha) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))+
      k^(alpha-1) * k * alpha * (delta-1) * beta^2 / ((1+n)*(1+n+d)*(1+beta+beta^2))+
      k * d / ((1+n+d)*(1+beta+beta^2))+
      k^alpha / ( 1- delta + alpha * k^(alpha - 1)) * (-d)/ ((1+n+d)*(1+beta+beta^2))
  )
}


get.path <- function(alpha = 0.5,
                     beta = 0.8,
                     d = 1,
                     n = 0.01,
                     delta = 0.9,
                     Time =50,
                     capital0 = 0.1,
                     tol = 0.01,
                     eta = 0.9,
                     max.iter = 50){
  
  capital_ss <- uniroot(capital.eq, c(1/10^100, 1),
                        tol = 1e-15,
                        alpha = alpha,
                        beta = beta,
                        d = d,
                        n = n,
                        delta = delta)$root
  print(capital_ss)
  tol <- capital_ss*tol
  
  # labour force ——
  L <- N <- rep(NA, Time)
  N[1] <- 1
  for(t in 2:Time){
    N[t] <- N[1] * (1+n)^(t-1)
  }
  
  for(t in 1:Time){
    L[t] <- N[1] * (1+n+d) * (1+n)^(t-2)
  }
  
  capital <- rep(NA, Time)
  
  
  
  # linear guess
  
  capital[1] <- capital0
  
  for(t in 2:Time){
    capital[t] <- capital[t-1] + (capital_ss-capital[1])/(Time-1)
  }
  
  error <- 1
  step <- 1
  # paths <- data.frame(step = integer(),
  # s_young=numeric(),
  # s_medium=numeric(),
  # c_young=numeric(),
  # c_medium=numeric(),
  # c_old=numeric(),
  # w=numeric(),
  # R=numeric(),
  # capital=numeric() )
  paths <- data.frame(step = 0,
                      alpha = alpha,
                      beta = beta,
                      d = d,
                      n = n,
                      delta = delta,
                      capital0 = capital0,
                      t = 1:Time,
                      s_young=NA,
                      s_medium=NA,
                      c_young=NA,
                      c_medium=NA,
                      c_old=NA,
                      w=NA,
                      R=NA,
                      capital=capital)
  
  while(error > tol | step > max.iter){
    paths <- rbind(paths,
                   update.path(capital = paths$capital[which(paths$step == step - 1)],
                               step=step, eta = eta,
                               alpha = alpha,
                               capital0 = capital0,
                               beta = beta,
                               d = d,
                               n = n,
                               delta = delta,
                               Time = Time))
    if(step > 2){
      error <- sqrt(sum((paths$capital[which(paths$step == step)] -
                           paths$capital[which(paths$step == (step-1))])^2))/Time
    }
    step <- step+1
  }
  
  return(list(L = L,
              N = N,
              capital_linear=capital,
              paths=paths))
}
out <- c(0.1, 0.05, 0.001) %>%
  map_dfr(function(capital0){get.path(tol = 0.00001, eta = 0.8, capital0 = capital0, delta = 1, Time =30)}$paths)
ggplot(out)+
  geom_line(aes(x=t, y = capital, alpha = factor(step),
                color = factor(capital0)), show.legend = FALSE)
ggplot(na.omit(out$paths))+
  geom_line(aes(x=t, y = c_young, alpha = factor(step)), show.legend = FALSE)


out <- expand.grid(alpha = c(0.4, 0.5, 0.6),
                   beta = c(0.9, 0.5, 0.1),
                   d = c(0.5, 1, 10),
                   n = c(0, 0.01),
                   delta = c(0.5, 1),
                   capital0 = 0.001) %>%
  map_dfr(function(capital0){get.path(tol = 0.00001,
                                      eta = 0.8,
                                      alpha = x$alpha,
                                      beta = x$beta,
                                      d = x$d,
                                      n = x$n,
                                      delta = x$delta,
                                      capital0 = x$capital0,
                                      Time =30)}$paths)
#rm(list=ls())

