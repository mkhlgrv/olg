# по статье demo_simulation
simulate_demography <- function(type=c("death", "birth", "migration"),
                                value_column="Female",
                                simul_time,n_components=6,
                                estimate="mean"){
  type <- match.arg(type)
  input_file <- paste0(type,'_rate.txt')
  input_table <-  read.csv(input_file) 
  if(type%in%c("migration","death","birth")){
    input_table <- input_table %>%
      filter(Year>1990)
  }
  if(type=="birth"){
    value_column <- "ASFR"
  }
  input_table <- input_table %>% dcast(Age~Year,value.var = value_column)
  input_table$Age[input_table$Age=="110+"] <- 111
  input_table <- input_table[order(as.numeric(input_table$Age)),]
  input_table[is.na(input_table)] <- 1
  if(type=="birth"){
    input_matrix <- 5*(input_table[,-1])^0.2-5
  } else if(type=="death"){
    input_matrix = log(input_table[,-1])
  } else{
    input_matrix <- input_table[,-1]
  }
  
  n_G <- nrow(input_matrix)
  S <- matrix(0, n_G, ncol(input_matrix))
  for(i in 2:ncol(input_matrix)){
    S[,i-1] <- spline(input_matrix[,i],n = n_G)$y
  }
  
  mu_S = rowMeans(S)
  interpolation_error = input_matrix[,-1]-mu_S
  
  pca <- prcomp(t(interpolation_error))
  Phi <- matrix(0, nrow=n_components, ncol= n_G)
  for(i in 1:n_G){
    y <- as.numeric(interpolation_error[i,])
    Phi[,i] <- lm(y~0+pca$x[,1:n_components])$coef
  }
  
  beta <- matrix(0, simul_time, n_components)
  for(j in 1:n_components){
    model <- forecast::ets(pca$x[,j])
    if(estimate=="mean"){
      beta[,j] <- forecast::forecast(model, simul_time)$mean
      
    } else if(estimate=="lower"){
      beta[,j] <- forecast::forecast(model, simul_time)$lower[,1]
    } else if(estimate=="upper"){
      beta[,j] <- forecast::forecast(model, simul_time)$upper[,1]
    }
    # источник дисперсии!
  }
  result_rate <- matrix(0, simul_time, n_G)
  for(i in 1:simul_time){
    result_rate[i,] <- mu_S+colSums(Phi*beta[i,])
  }
  if(type=="birth"){
    rate_detransformed <- ((result_rate+5)/5)^5
  } else if(type=="death"){
    rate_detransformed <- exp(result_rate)
  } else{
    rate_detransformed <- result_rate
  }
  
  list("rate"=rate_detransformed, "Phi"=Phi, "beta"=beta)
}
result_rate <- simulate_demography("migration","Female",200,n_components = 6,"upper")
ggplot(melt(result_rate$rate[c(1,10,50,100,200,200,200),1:110]))+
  geom_line(aes(x=Var2, y = value, group=Var1), alpha=0.2)

result_rate <- simulate_demography("birth",simul_time=200,n_components = 6, "lower")
ggplot(melt(result_rate$rate[c(1,10,50,100,200,200,200),1:40]))+
  geom_line(aes(x=Var2, y = value, group=Var1), alpha=0.2)

result_rate <- simulate_demography("death","Male",simul_time=200,n_components = 6, "upper")
ggplot(melt(result_rate$rate[c(1,2,10,50,100,200,200,200),1:80]))+
  geom_line(aes(x=Var2, y = value, group=Var1), alpha=0.2)



sum(rev(demographics_data$N_steady)*(((result_rate[200,]+2)/2)^2)[-c(1:15)])/sum(demographics_data$N_steady)

res <- get.demographics(96,100,eps = c(rep(1, 50), rep(0, 46)), FALSE)
time_n <- length(res$demography_rates$mortality_rate_total)
ggplot(data=NULL,aes(x=1:time_n))+
  geom_line(aes(y = res$demography_rates$mortality_rate_total,color="Death rate"))+
  geom_line(aes(y = res$demography_rates$birth_rate_total, color= "Birth rate"))
res_migration <- get.demographics(96,100,eps = c(rep(1, 50), rep(0, 46)), TRUE)
ggplot(data=NULL,aes(x=1:time_n))+
  geom_line(aes(y = res_migration$demography_rates$migration_rate_total_male,color="Migration male"))+
  geom_line(aes(y = res_migration$demography_rates$migration_rate_total_female, color= "Migration female"))

# население
(rowSums(res$demography_rates$N))

ggplot(melt(res$demography_rates$Fertility[c(1,10,50,100,200,200,200),1:55]))+
  geom_line(aes(x=Var2, y = value, group=Var1), alpha=0.2)

# распределение возрастов
ggplot(melt(res$demography_rates$Fertility[c(1,10,50,100,200,200,200),1:55]))+
  geom_line(aes(x=Var2, y = value, group=Var1), alpha=0.2)
