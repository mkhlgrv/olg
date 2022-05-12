get.steady.state <- function(K_steady_0 = 20000, niter = 100, eta = 0.8, tol = 1/10^4, save_path=FALSE,
                             L_steady_0,
                             N_steady,
                             N,
                             simul_time,
                             eps,
                             eps_ret,
                             G,
                             Pi,
                             beta,
                             theta,
                             delta ,
                             alpha,
                             tau_pi,
                             tau_va,
                             tau_insurance,
                             tau_retirement,
                             tau_i){
  k_steady <- K_steady_0/L_steady
  k_new <- 0

  l <- 1
  path <- NULL
  if(save_path){
    path <- tibble::tibble(iter =integer(), w = numeric(), r=numeric())
  }
  
  while((l <= niter) & (abs((k_steady-k_new)/k_steady)>tol)){
    
    


    
    # r_steady <- rent.rate(k_steady, alpha, delta, tau_pi)
    r_steady <- 0.01
    price_n 
    
    r_steady_tax_adjusted <- rent.rate.tax.adjusted(r_steady, tau_i)
    w_steady <- wage.rate(k_steady, alpha)
    
    # retirement rate

    ret_steady <- retirement.rate(t=simul_time,
                                  w_steady, N=N,G=G,
                                  eps=eps, eps_ret=eps_ret,
                                  tau_retirement=tau_retirement,
                                  tau_insurance=tau_insurance)
    
    get.bequest(t=simul_time,g = G,G=G,N =N,Mortality = )
    
  
    
    cons_init <- cons.initial(g = G, t =1,Pi=Pi,
                              r_tax_adjusted = r_steady_tax_adjusted,
                              w = w_steady,
                              ret = ret_steady,
                              a_init = 0,
                              G=G,
                              eps=eps, eps_ret=eps_ret,
                              tau_i=tau_i,
                              tau_insurance=tau_insurance,
                              tau_va=tau_va,
                              tau_retirement=tau_retirement,
                              beta=beta,
                              theta=theta,
                              steady = TRUE)
    
    cons_steady <- cons.recursive(g =G, t = 1, cons_init = cons_init, r_tax_adjusted = r_steady_tax_adjusted,
                                  G =G, Pi=Pi,
                                  beta=beta,
                                  theta=theta, 
                                  steady = TRUE)
    
    assets_steady <- asset.recursive(g =G,G=G, t = 1,cons = cons_steady, a_init = 0, w = w_steady, r_tax_adjusted = r_steady_tax_adjusted, ret = ret_steady,
                                     eps=eps, eps_ret=eps_ret,tau_va=tau_va, tau_insurance=tau_insurance, tau_retirement=tau_retirement, tau_i=tau_i,
                                     steady = TRUE)
    
    

    k_new <- sum(assets_steady*rev(N_steady))/L_steady

    k_steady <- eta * k_steady + (1-eta)* k_new
    
    if(save_path){
      path <- path %>% add_row(iter =l, w = w_steady, r=r_steady)
    }
    
    l <- l+1

    
  }
  return(list(w = w_steady, r = r_steady, k = k_steady, K = k_steady*L_steady, cons = cons_steady, assets= assets_steady, ret = ret_steady, path = path))
}


#
cons.initial <- function(g, t,r_tax_adjusted,w, ret, a_init,G,
                         Pi,
                         eps, eps_ret,
                         tau_i,
                         tau_retirement,
                         tau_insurance,
                         tau_va,
                         beta,
                         theta,
                         steady = TRUE){
  if(g<t|g>G+t-1){
    return(0)
  } else{
    if(steady){
      r_tax_adjusted <- rep(r_tax_adjusted, g)
      w <- rep(w, g)
      ret <- rep(ret, g)
    }
    
    numenator <- a_init+
      sum(
        (eps_ret[(G-g+t):G]*ret[t:g]+
           (1-tau_i)*
           (1- (tau_retirement+tau_insurance)/
              (1+tau_retirement+tau_insurance))*eps[(G-g+t):G]*w[t:g])
        *cumprod(1/(1+r_tax_adjusted[t:g]))
      )
    
    
    denominator <- (1+tau_va)/(1+r_tax_adjusted[t])
    if(g != t){

      
      denominator <- denominator+
        sum((1+tau_va)*(beta^(seq(1, g-t, by = 1))*Pi[(t-1)*(G-1)+seq(1, g-t, by = 1), G-g+t]
                        *cumprod(1+r_tax_adjusted[(t+1):g]))^(1/theta)/
              (cumprod(1+r_tax_adjusted[t:g])[-1]))
    }
  }
  numenator/denominator
  
  
  
}


cons.recursive <- function(cons_init, r_tax_adjusted, g, t,
                           beta, theta,Pi, G,
                           steady = TRUE){
  if(steady){
    r_tax_adjusted <- rep(r_tax_adjusted, g)
  }
  if(t+1>g){
    cons_init
  } else{
    cons_init* c(1, (beta^(seq(1, g-t, by = 1))*
                       Pi[(t-1)*(G-1)+seq(1, g-t, by = 1), G-g+t]*
                       cumprod(1+ r_tax_adjusted[(t+1):g]))^(1/theta))
  }
}



asset.recursive <- function(cons, a_init,w,  r_tax_adjusted, ret, g,G, t,
                            tau_va,
                            eps, eps_ret, tau_insurance, tau_retirement, tau_i,
                            steady = TRUE
){
  if(steady){
    r_tax_adjusted <- rep(r_tax_adjusted, g)
    w <- rep(w, g)
    ret <- rep(ret, g)
  }
  
  assets <- a_init*(1+r_tax_adjusted[t])+
    eps[G-g+t]*w[t]*
    (1-(tau_insurance+tau_retirement)/(1+tau_insurance+tau_retirement))*(1-tau_i) +
    eps_ret[G-g+t]*ret[t] -
    cons[1]*(1+tau_va)
  
  if(g>=(t+1)){
    for(i in (t+1):g){
      assets[i-t+1] <-assets[i-t]*(1+r_tax_adjusted[i])+
        eps[G-g+i]*w[i]*
        (1-(tau_insurance+tau_retirement)/(1+tau_insurance+tau_retirement))*(1-tau_i) +
        eps_ret[G-g+i]*ret[i] -
        cons[i-t+1]*(1+tau_va)
    }
  }
  
  assets
  
}




rent.rate <- function(k, alpha, delta, tau_pi){
  (alpha*k^(alpha-1) - delta)*(1-tau_pi)
}



rent.rate.tax.adjusted <- function(r, tau_i){
  r*(1-tau_i)
}




wage.rate <- function(k, alpha){
  (1-alpha)* k^alpha
}


retirement.rate <- function(t, w, N,G, eps, eps_ret, tau_retirement, tau_insurance){

  tau_retirement/(1+tau_retirement+tau_insurance)*w*sum(rev(N[t, t:(G+t-1)])*eps)/sum(rev(N[t, t:(G+t-1)])*eps_ret)
}


get.assets.initial <- function(N_steady, N, assets, K_1, G){
  # под N_steady понимаем N в периоде, который предшествует steady state
  # (потому что определение уровня накопленногок апитала происходит в прошлом периоде времени)
  # последний элемент должен быть равен 0, но не будет равен в точности 0 из-за компьютерных вычислений
  
  assets[G] <- 0
  

  share_init <- (assets*rev(N_steady))/(sum(assets*rev(N_steady)))
  
  share_init <- c(0, share_init[1:(G-1)])
  
  share_init/rev(N[1,1:G])*K_1
  
}
get.path.to.ss <- function(ss,k_1, simul_time,
                           tol,
                           eta,
                           L,
                           N,
                           N_steady,
                           nsteps,
                           eps,
                           eps_ret,
                           G,
                           Pi,
                           beta,
                           theta,
                           delta,
                           alpha,
                           tau_pi,
                           tau_va,
                           tau_insurance,
                           tau_retirement,
                           tau_i){
  
  
  
  #  
  # симулируем не только simul_time, но еще G-1 лет после достижения ss
  # для того, чтобы все поколения, год рождения которых <=simul_time
  # а их будет ровно simul_time+G-1
  # могли принять решение, зная будущее
  # +1 и -1 потому что последняя точка первого массива это уже ss
  
  
  k <- c(seq(k_1, ss$k, length.out = simul_time + 1), rep(ss$k, G-1-1))
  
  
  k_history <- matrix(data = k, nrow = 1, ncol = simul_time+G-1)
  
  # factor prices
  
  w <- c(rep(0, simul_time), rep(ss$w, G-1))
  r <- c(rep(0, simul_time), rep(ss$r, G-1))
  # retirement system
  ret <- c(rep(0, simul_time), rep(ss$ret, G-1))
  
  
  
  
  # Consumption, Assets
  
  # строка это год, столбец - номер поколения
  Cons <- matrix(0, nrow = simul_time+G-1, ncol = simul_time+G-1 )
  A <- matrix(0, nrow = simul_time+G-1, ncol = simul_time+G-1 )
  
  
  # intitial endowment
  # у поколения под номером G его нет,
  # а поколения с 1 по G-1 получают endowment в виде доли общего запаса капитала k_1*L[1] в момент времени t=1
  # которая равна доли, которую они имеют в steady state в конце t=0 (при учете их численности)
  
  # в массиве ss$assets[i] это то, сколько поколение возраста [i] сберегло на начало года
  A_initial <- rep(0, simul_time+G-1)
  
  A_initial[1:G] <- rev(get.assets.initial(N_steady = N_steady,N = N, assets = ss$assets, K_1 = k_1*L[1], G = G))
  
  
  step <- 1
  k_new <- rep(0, simul_time+G-1)
  
  while(step <= nsteps & max(abs((k - k_new)/ k)) > tol){
    r <- rent.rate(k, alpha, delta, tau_pi)
    r_tax_adjusted <- rent.rate.tax.adjusted(r, tau_i)
    w <- wage.rate(k, alpha)
    
    
    
    
    # retirement rate
    for(i in 1:simul_time){
      ret[i] <- retirement.rate(t = i,
                                w=w[i],N=N,G=G,
                                eps=eps, eps_ret=eps_ret, tau_retirement=tau_retirement, tau_insurance=tau_insurance)
      
    }
    
    
    
    # # временно
    # 
    # w[1:3] <- 1
    # ret[1:3] <- 1
    # r_tax_adjusted[1:3] <- 0
    # A_initial[1:3] <- 1
    # # потом убрать
    
    
    for(g in 1:(simul_time+G-1)){
      # время, с которого начинаем считать выбор индивида из поколения g
      # g -это номер поколения и одновременно момент смерти
      start_time <-  max(1, g-G+1)
      
      cons_init <- cons.initial(g = g, t = start_time, r_tax_adjusted = r_tax_adjusted,
                                w = w, ret = ret, a_init = A_initial[g],G=G,
                                Pi=Pi,
                                eps=eps, eps_ret=eps_ret,
                                tau_i=tau_i,
                                tau_insurance=tau_insurance,
                                tau_va=tau_va,
                                tau_retirement=tau_retirement,
                                beta=beta,
                                theta=theta,
                                steady = FALSE)
      
      Cons[start_time:g,g] <- cons.recursive(cons_init=cons_init,
                                             r_tax_adjusted = r_tax_adjusted, g = g, t = start_time, beta=beta,
                                             G =G, Pi=Pi,
                                             theta=theta,
                                             steady = FALSE)
      
      
      A[start_time:g,g] <- asset.recursive(cons = Cons[start_time:g,g],
                                           a_init = A_initial[g], w = w,
                                           r_tax_adjusted = r_tax_adjusted, ret = ret, g = g,G=G, t = start_time,
                                           eps=eps,
                                           eps_ret=eps_ret,
                                           tau_insurance=tau_insurance, tau_retirement=tau_retirement, tau_i=tau_i,tau_va=tau_va,steady = FALSE)
      
    }
    
    # обновляем guess капитала, при этом
    # активы, сохраненные в момент времени t=simul_time, мы не суммируем: мы предполагаем, что они уже будут находится на уровне steady state
    # тк сумма активов(t=simul_time) = K(t=simul_time+1) = K_ss
    # делим на численность рабочей силы в следуующем периоде!
    # k_new длины simul_time+G-1 --- равна длине k
    k_new <- c(k_1,
               rowSums(A[1:(simul_time-1),1:(simul_time+G-2)]*N[1:(simul_time-1),1:(simul_time+G-2)])/L[2:(simul_time)],
               rep(ss$k, G-1))
    
    # A_total_mat <- A[1:(simul_time-1),1:(simul_time+G-2)]*N[1:(simul_time-1),1:(simul_time+G-2)]
    # debt_share <- rep(NA, nrow(A_total_mat))
    # if(step == 5){
    #   for(i in 1:nrow(A_total_mat)){
    #     debt <- A_total_mat[i,] %>% which(x = .<0) %>% sum %>% abs
    #     debt_share[i] <- debt/(sum(A_total_mat[i,]) + debt)
    #   }
    #   print(plot(debt_share, type='l'))
    #   stop()
    # }

    
    
    
    
    
    k <- eta * k + (1-eta)  * k_new
    step <- step+1
    
    k_history <- rbind(k_history, k)
    
  }
  
  
  return(list(k=k, A = A, Cons = Cons, k_history=k_history))
  
}


get.demography.rates <- function(G_with_children,simul_time, use_migration){
  female_to_male_ratio <- 0.9489044
  female_ratio <- female_to_male_ratio/(1+female_to_male_ratio)
    pop_size_female <- read.csv('pop_size_corrected.txt')  %>%
      filter(Year==max(Year)) %>% .[,"Female"] %>%
      .[1:G_with_children]
    pop_size_male <- read.csv('pop_size_corrected.txt')  %>%
      filter(Year==max(Year)) %>% .[,"Male"] %>%
      .[1:G_with_children]
    
    

    mortality_rate_female <- simulate_demography("death","Female",simul_time=simul_time+G_with_children-1,
                                                 n_components = 5,
                                                 lambda = 5, "mean",
                                                 start_year = 1960, end_year = 2014
                                                 )$rate
    mortality_rate_male <- simulate_demography("death","Male",simul_time=simul_time+G_with_children-1,
                                               n_components = 5,
                                               lambda = 5, "mean",
                                               start_year = 1960, end_year = 2014)$rate
    
    if(use_migration){
      migration_rate_female <- simulate_demography("migration", "Female",simul_time=simul_time+G_with_children-1,
                                                   n_components = 5,
                                                   lambda = 0, "mean",
                                                   start_year = 1960, end_year = 2014)$rate
      migration_rate_male <- simulate_demography("migration", "Male",simul_time=simul_time+G_with_children-1,
                                                 n_components = 5,
                                                 lambda = 0, "mean",
                                                 start_year = 1960, end_year = 2014)$rate
    } else{
      migration_rate_male <- migration_rate_female <-
        matrix(0,nrow(mortality_rate_female), ncol(mortality_rate_female))
      
    }
    
    mortality_rate_female[,G_with_children] <- 1
    mortality_rate_male[,G_with_children] <- 1
    
    fertility_rate_birth_age <- simulate_demography("birth",simul_time=simul_time+G_with_children-1,
                                                    n_components = 5,
                                                    lambda = 5, "mean",
                                                    start_year = 1960, end_year = 2014
    )$rate
    fertility_rate <- matrix(0, nrow(mortality_rate_female), ncol(mortality_rate_female))
    
    fertility_rate[,12:55] <- fertility_rate_birth_age
    
    # +G_with_children-1 потому что нужно для расчета survival probability
    Migration_female <- Migration_male <- Mortality_female <- Mortality_male <-  Fertility <- N_female <- N_male <- N <- matrix(0, nrow=simul_time+G_with_children-1,
                                                                                           ncol = simul_time+G_with_children-1+G_with_children-1)
    
    for(i in 1:simul_time){
      Mortality_female[i, i:(i+G_with_children-1)] <- rev(mortality_rate_female[i,])
      Mortality_male[i, i:(i+G_with_children-1)] <- rev(mortality_rate_male[i,])
      
      Migration_female[i, i:(i+G_with_children-1)] <- rev(migration_rate_female[i,])
      Migration_male[i, i:(i+G_with_children-1)] <- rev(migration_rate_male[i,])
      
      Fertility[i, i:(i+G_with_children-1)] <- rev(fertility_rate[i,])
    }
    for(i in (simul_time+1):(simul_time+G_with_children-1)){
      Mortality_female[i, i:(i+G_with_children-1)] <- rev(mortality_rate_female[simul_time,])
      Mortality_male[i, i:(i+G_with_children-1)] <- rev(mortality_rate_male[simul_time,])
      
      Migration_female[i, i:(i+G_with_children-1)] <- rev(migration_rate_female[simul_time,])
      Migration_male[i, i:(i+G_with_children-1)] <- rev(migration_rate_male[simul_time,])
      
      Fertility[i, i:(i+G_with_children-1)] <- rev(fertility_rate[simul_time,])
    }
    
    N_female[1, 1:G_with_children] <- rev(pop_size_female/1000000)
    N_male[1, 1:G_with_children] <- rev(pop_size_male/1000000)
    N[1, 1:G_with_children] <- N_female[1, 1:G_with_children]+N_male[1, 1:G_with_children]
    
    fertility_rate_total <- mortality_rate_total_female<-mortality_rate_total_male <-
      mortality_rate_total <- migration_rate_total_female <- migration_rate_total_male<- 
      migration_rate_total <- birth_rate_total <-  rep(NA, simul_time)
    
    for(t in 2:nrow(N)){
      # deaths and migration
      N_female[t,(t-1):(G_with_children+t-2)] <- N_female[t-1,(t-1):(G_with_children+t-2)]*
        (1- Mortality_female[t-1,(t-1):(G_with_children+t-2)])*(1+ Migration_female[t-1,(t-1):(G_with_children+t-2)])
      mortality_rate_total_female[t-1] <-
        sum(N_female[t-1,(t-1):(G_with_children+t-2)]*
              Mortality_female[t-1,(t-1):(G_with_children+t-2)])/sum(N_female[t-1,(t-1):(G_with_children+t-2)])
      migration_rate_total_female[t-1] <-
        sum(N_female[t-1,(t-1):(G_with_children+t-2)]*
              Migration_female[t-1,(t-1):(G_with_children+t-2)])/sum(N_female[t-1,(t-1):(G_with_children+t-2)])
      
      N_male[t,(t-1):(G_with_children+t-2)] <- N_male[t-1,(t-1):(G_with_children+t-2)]*
        (1- Mortality_male[t-1,(t-1):(G_with_children+t-2)])*(1+ Migration_male[t-1,(t-1):(G_with_children+t-2)])
      mortality_rate_total_male[t-1] <-
        sum(N_male[t-1,(t-1):(G_with_children+t-2)]*
              Mortality_male[t-1,(t-1):(G_with_children+t-2)])/sum(N_male[t-1,(t-1):(G_with_children+t-2)])
      migration_rate_total_male[t-1] <-
        sum(N_male[t-1,(t-1):(G_with_children+t-2)]*
              Migration_male[t-1,(t-1):(G_with_children+t-2)])/sum(N_male[t-1,(t-1):(G_with_children+t-2)])
      
      N[t,(t-1):(G_with_children+t-2)] <- N_female[t,(t-1):(G_with_children+t-2)]+N_male[t,(t-1):(G_with_children+t-2)]
      
      mortality_rate_total[t-1] <- (sum(N_female[t-1,(t-1):(G_with_children+t-2)]*
                                     Mortality_female[t-1,(t-1):(G_with_children+t-2)])+
                                 sum(N_male[t-1,(t-1):(G_with_children+t-2)]*
                                       Mortality_male[t-1,(t-1):(G_with_children+t-2)]))/sum(N[(t-1),(t-1):(G_with_children+t-2)])
      
      migration_rate_total[t-1] <- (sum(N_female[t-1,(t-1):(G_with_children+t-2)]*
                                          Migration_female[t-1,(t-1):(G_with_children+t-2)])+
                                 sum(N_male[t-1,(t-1):(G_with_children+t-2)]*
                                       Migration_male[t-1,(t-1):(G_with_children+t-2)]))/sum(N[(t-1),(t-1):(G_with_children+t-2)])
      # births

      N_female[t, G_with_children+t-1] <- female_ratio*sum(N_female[t-1,(t-1):(G_with_children+t-2)]*
                                                             Fertility[t-1,(t-1):(G_with_children+t-2)])
      N_male[t, G_with_children+t-1] <- (1-female_ratio)*sum(N_female[t-1,(t-1):(G_with_children+t-2)]*
                                                               Fertility[t-1,(t-1):(G_with_children+t-2)])
      N[t, G_with_children+t-1] <- N_female[t, G_with_children+t-1]+N_male[t, G_with_children+t-1]
      fertility_rate_total[t-1] <- sum(N_female[t-1,(t-1):(G_with_children+t-2)]*
                                         Fertility[t-1,(t-1):(G_with_children+t-2)])/
        sum(N_female[t-1,(t-1):(G_with_children+t-2)])
      birth_rate_total[t-1] <- sum(N_female[t-1,(t-1):(G_with_children+t-2)]*
                                     Fertility[t-1,(t-1):(G_with_children+t-2)])/
        (sum(N[(t-1),(t-1):(G_with_children+t-2)]))
    }
    list(Mortality_female=Mortality_female,
         Mortality_male=Mortality_male,
         Fertility=Fertility,
         N_female=N_female,
         N_male=N_male,
         N=N,
         fertility_rate_total=fertility_rate_total, 
         mortality_rate_total_female=mortality_rate_total_female, 
         mortality_rate_total_male=mortality_rate_total_male,
         mortality_rate_total=mortality_rate_total, 
         birth_rate_total=birth_rate_total,
         migration_rate_total_female=migration_rate_total_female,
         migration_rate_total_male=migration_rate_total_male,
         migration_rate_total=migration_rate_total_male)
         
}

get.demographics <- function(G, simul_time, eps,use_migration){

    
  G_with_children <- G+15
  demography_rates <- get.demography.rates(G_with_children,simul_time,use_migration)
  N_female <- demography_rates$N_female
  N_male <- demography_rates$N_male
    for(i in 1:nrow(N_male)){
      for(j in 1:ncol(N_male)){


        age <- G_with_children-j+i
        if(age<16){
          # N_female[i,j] <- 0

        }
      }
    }
    crimea_coef <- 146.267288/rowSums(N_female[1:2,]+N_male[1:2,])[2]
    
    N_female <- crimea_coef*N_female
    N_male <- crimea_coef*N_male
    
    N_female_steady <- N_female[(simul_time-1),(simul_time-1):(simul_time-1+G-1)]
    N_male_steady <- N_male[(simul_time-1),(simul_time-1):(simul_time-1+G-1)]
    
    Pi_female <- get.survival.probability(G = G, simul_time = simul_time, 
                                   N = get.demography.rates(G_with_children,simul_time,FALSE)$N_female)
    Pi_male <- get.survival.probability(G = G, simul_time = simul_time, 
                                  N = get.demography.rates(G_with_children,simul_time,FALSE)$N_male)
    
    
    Pi_female_steady <- Pi_female[((simul_time-1)*(G-1)+1):((simul_time)*(G-1)),]
    Pi_male_steady <- Pi_male[((simul_time-1)*(G-1)+1):((simul_time)*(G-1)),]
    list(demography_rates=demography_rates,
         N_female=N_female,
         N_male=N_male,
         N_female_steady=N_female_steady,
         N_male_steady=N_male_steady,
         Pi_female = Pi_female,
         Pi_male=Pi_male,
         Pi_female_steady=Pi_female_steady,
         Pi_male_steady=Pi_male_steady)

  
}

get.labour.force <- function(N, eps = c(rep(1, G-20), rep(0, 20)), simul_time=200, G=60){
  # labor force
  L <- rep(0, simul_time)
  for(t in 1:length(L)){
    
    L[t] <- sum(rev(eps)*N[t,t:(t+G-1)])
  }
  L
}


get.survival.probability <- function(G, simul_time, N){
  Pi <- matrix(data=0, nrow=simul_time*(G-1), ncol = G)
  for(m in 1:nrow(Pi)){
    added_time <- mod(m, G-1) %>% ifelse(test =( .==0), yes = G-1,no =  .) 
    time_1 <- (m %/% (G-1)) %>% ifelse(test = (added_time==G-1),
                                       yes = .,
                                       no =  sum(.,1))
    
    time_2 <- time_1 + added_time
    
    
      # оцениваем вероятность выжить до периода time_2, если агент жив в периоде time_1
      
      # 1-ое поколение - это самое молодое
      # важно: N без миграции
    
      Pi[m,1:G] <- rev(N[time_2, (time_1):(G-1+time_1)]/N[time_1, (time_1):(G-1+time_1)])
  }
  Pi
}

get.bequest <- function(t,
                        g,
                        G,
                        N,
                        Mortality,
                        Fertility,
                        Pi,
                        asset){
  # asset на первом месте самое молодое поколение
  G_with_children <- G+15
  # 
  
   adult_children <- rep(0,g-t+1)
  for(j in (t-1):(g-1)){
    adult_children[j-t+2] <- sum(Fertility[(j-G_with_children+1):(t-16),j]* # cohort j, time i from i=(j-G_with_children+1) to (t-16)
          N[(j-G_with_children+1):(t-16),j])
  }

  # G_with_children <- G
  # for(j in (t-1):(g-1)){
  #   adult_children[j-t+2] <- sum(Fertility[(j-G_with_children+1):(t-1), j]* # cohort j, time i from i=(j-G_with_children+1) to (t-16)
  #                                  N[(j-G_with_children+1):(t-1),j])
  # }

   # print(Mortality[t-1,
   #                 (t-1):(g-1)])
  sum(N[t-1,
        (t-1):(g-1)]*
        Mortality[t-1,
                    (t-1):(g-1)]* # в момент времени t-1
        asset[t-1,
                  (t-1):(g-1)]*
        Fertility[g-G_with_children,
                           (t-1):(g-1)]/ # в момент времени g-G_with_children+1
      adult_children
      ) 

  
}

master.function <- function(G = 96,
                              simul_time = 500,
                              theta = 1,
                              
                              beta = 0.99,
                              delta = 0.03,
                              alpha = 0.4,
                              k_initial=1,
                              
                              tau_pi =  0.2,
                              tau_va = 0.2,
                              tau_insurance = 0.08,
                              tau_retirement = 0.22,
                              tau_i = 0.13,
                              
                              # age-specified productivity
                              eps = c(rep(1, 50), rep(0, 46)),
                              # retirement coefficient
                              eps_ret = c(rep(0, 50), rep(1, 46)),
                              
                              
                              K_steady_0 = 10,
                              niter_steady = 1000,
                              eta_steady = 0.7,
                              tol_steady = 1/10^5,
                              save_path_steady = FALSE,
                              
                              niter_path = 100,
                              eta_path = 0.8,
                              tol_path = 1/10^5,
                            
                            save.demographics = FALSE,
                            update_demographics=TRUE
                              
){
  
  
  demographics <- get.demographics(G = G,
                                   simul_time = simul_time,
                                   eps = eps,update = update_demographics
                                  
                                    )


  ss <- get.steady.state(K_steady_0 = K_steady_0,
                         niter = niter_steady,
                         eta = eta_steady,
                         tol = tol_steady,
                         L_steady=demographics$L_steady,
                         N_steady=demographics$N_steady,
                         N=demographics$N,
                         simul_time = simul_time,
                         
                         eps=eps,
                         eps_ret=eps_ret,
                         G=G,
                         Pi=demographics$Pi_steady,
                         beta=beta,
                         theta=theta,
                         delta = delta,
                         alpha = alpha,
                         tau_pi=tau_pi,
                         tau_va=tau_va,
                         tau_insurance=tau_insurance,
                         tau_retirement=tau_retirement,
                         tau_i=tau_i,
                         save_path = save_path_steady)
  
  out <- get.path.to.ss(ss = ss,
                        k_1 = k_initial,
                        simul_time = simul_time,
                        nsteps =niter_path,
                        tol = tol_path,
                        eta = eta_path,
                        L=demographics$L,
                        N=demographics$N,
                        N_steady=demographics$N_steady,
                        Pi=demographics$Pi,
                        eps=eps,
                        eps_ret=eps_ret,
                        G=G,
                        beta=beta,
                        theta=theta,
                        delta = delta,
                        alpha = alpha,
                        tau_pi=tau_pi,
                        tau_va=tau_va,
                        tau_insurance=tau_insurance,
                        tau_retirement=tau_retirement,
                        tau_i=tau_i)
  if(save_path_steady){
    out <- c(out,path= list(ss$path)) 
  }
  if(save.demographics){
    out <- c(out,demographics= list(demographics))
  }
  out
  
  
}



