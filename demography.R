# по статье demo_simulation
simulate_demography <- function(type=c("death", "birth", "migration"),
                                value_column="Female",
                                simul_time,n_components=6,lambda =1,start_year=1980,end_year=2014,
                                estimate="mean"){
  type <- match.arg(type)
  if(type%in%c("death", "birth")){
    input_file <- paste0(type,'_rate_corrected.txt')
    input_table <-  read.csv(input_file)
  } else{
    input_file <- "migration_rate.txt"
    input_table <-  read.csv(input_file)
  }
  
  

  if(type=="birth"){
    value_column <- "ASFR"
  }

  
  if (is.null(start_year)){
    if(type%in%c("migration","death","birth")){
      if (value_column=="Male"){
        start_year = 1980
      } else if (type == "birth"){
        start_year = 1980
      }
    }
  }

  input_table <- input_table %>% filter(Year >=start_year & Year <=end_year)
  
  # View(input_table)
  input_table <-  reshape2::dcast(input_table,Age~Year,value.var = value_column)
  
  input_table <- input_table[order(as.numeric(input_table$Age)),]
  input_table[is.na(input_table)] <- 1
  
  # if(type=="birth"){
  #   input_matrix <- 5*(input_table[,-1])^0.2-5
  # } else if(type=="death"){
  #   input_matrix = log(input_table[,-1])
  # } else{
  #   input_matrix <- input_table[,-1]
  # }

  
  if(lambda==1){
    input_matrix <- log(input_table[,-1])
  } else if(lambda==0){
    input_matrix <- input_table[,-1]
  }
  else {
    input_matrix <- lambda*(input_table[,-1])^(1/lambda)-lambda
  }
  
  
  n_G <- nrow(input_matrix)
  ncol(input_matrix)
  
  
  # # death
  # if (type=="death"){
  #   x = seq(11,100,1)
  #   x_sq = x**2
  #   coefs <- matrix(data = NA, nrow = nrow(input_matrix)-11, ncol = 3)
  #   for (i in 1:ncol(input_matrix)){
  #     model <- lm(input_matrix[11:(nrow(input_matrix)-11),i]~x+x_sq)
  #     coefs[i,] <- model$coefficients
  #   }
  #   coefs_1_forecast = coefs[,1] %>%
  #     auto.arima() %>%
  #     forecast(h=h)
  #   coefs_2_forecast = coefs[,2] %>%
  #     auto.arima() %>%
  #     forecast(h=h)
  #   coefs_3_forecast = coefs[,3] %>%
  #     auto.arima() %>%
  #     forecast(h=h)
  #   
  #   result <- function (age,h){
  #     coefs_1_forecast$mean[h] +
  #       age*coefs_2_forecast$mean[h]+
  #       age**2*coefs_3_forecast$mean[h]
  #   }
  #   y <-  x
  #   for (i in 1:length(y)){
  #     y[i] <- result(x[i],h)
  #   }
  #   y_pred = exp(y)
  #   
  # }
  # 
  # if (type=="birth"){
  #   #birth
  #   x = seq(12,55,1)
  #   x_sq = x**2
  #   coefs <- matrix(data = NA, nrow = ncol(input_matrix), ncol = 3)
  #   for (i in 1:ncol(input_matrix)){
  #     model <- lm(input_matrix[,i]~x+x_sq)
  #     coefs[i,] <- model$coefficients
  #   }
  #   
  #   coefs_1_forecast = coefs[,1] %>%
  #     auto.arima() %>%
  #     forecast(h=h)
  #   coefs_2_forecast = coefs[,2] %>%
  #     auto.arima() %>%
  #     forecast(h=h)
  #   coefs_3_forecast = coefs[,3] %>%
  #     auto.arima() %>%
  #     forecast(h=h)
  #   
  #   result <- function (age,h){
  #     coefs_1_forecast$mean[h] +
  #       age*coefs_2_forecast$mean[h]+
  #       age**2*coefs_3_forecast$mean[h]
  #   }
  #   y <-  x
  #   for (i in 1:length(y)){
  #     y[i] <- result(x[i],h)
  #   }
  #   
  #   y_pred = round(((y+5)/5)**5,5)
  # }
  n_G <- nrow(input_matrix)
  S <- matrix(0, n_G, ncol(input_matrix))
  for(i in 2:ncol(input_matrix)){
      S[,i-1] <- smooth.spline(input_matrix[,i],n = n_G,df = 5)$y
    
  }

  mu_S = rowMeans(S)
  interpolation_error = input_matrix[,-1]-mu_S
  pca <- prcomp(t(interpolation_error))
  Phi <- matrix(0, nrow=(n_components+1), ncol= n_G)
  for(i in 1:n_G){
    y <- as.numeric(interpolation_error[i,])
    Phi[,i] <- lm(y~pca$x[,1:n_components])$coef
  }

  beta <- matrix(0, nrow=simul_time, ncol=n_components)
  
  for(j in 1:n_components){
    model <- forecast::ets(pca$x[,j])
    # if (j ==1){
      if(estimate=="mean"){
        beta[,j] <- forecast::forecast(model, simul_time)$mean
        
      } else if(estimate=="lower"){
        beta[,j] <- forecast::forecast(model, simul_time)$lower[,1]
      } else if(estimate=="upper"){
        beta[,j] <- forecast::forecast(model, simul_time)$upper[,1]
      }
    # } else{
    #   beta[,j] <- rep(pca$x[nrow(pca$x),j], simul_time)
    # }

  }
  projection <- matrix(0, simul_time, n_G)
  for(i in 1:100){

    projection[i,] <- mu_S+Phi[1,]+colSums(Phi[-1,]*beta[i,])
    
    if(type=="migration"){
      projection[i,] <- smooth.spline(projection[i,], df = 30)$y
    }
  }
  for (i in 101:simul_time){
    projection[i,] <- projection[100,]
  }
  
  
 
  
  if(lambda==1){
    projection <- exp(projection)
  } else if(lambda==0){
    projection <- projection
  }else{
    projection <- ((projection+lambda)/lambda)^lambda
  }
  rate_detransformed <- projection
  for(i in 1:simul_time){
    if(i<100){
      if(type=="migration"){
        rate_detransformed[i,] <- 0*projection[i,]+1*(i/100*projection[i,]+(1-i/100)*rowMeans(input_table[,(ncol(input_table)-10):(ncol(input_table)-1)]))
      }
    }

      
  }
  
  
  
  list("rate"=rate_detransformed, "Phi"=Phi, "beta"=beta, "input"=input_table, "pca"=pca$x, beta=beta)
}
# result_rate <- simulate_demography("migration","Female",200,n_components = 6,"upper")
# ggplot(melt(result_rate$rate[c(1,10,50,100,200,200,200),1:110]))+
#   geom_line(aes(x=Var2, y = value, group=Var1), alpha=0.2)

result_rate <- simulate_demography("birth",simul_time=600,n_components = 5,lambda = 5, "mean", start_year = 1960, end_year = 2014)
ggplot()+
  geom_line(data = melt(result_rate$rate[c(100,100),1:40]),aes(x=Var2, y = value, group=Var1), alpha=0.2)+
  geom_line(aes(1:40,result_rate$input[1:40,ncol(result_rate$input)]), color ="red")

result_rate <- simulate_demography("death","Male",simul_time=600,n_components = 5,lambda = 1, "mean", start_year = 1960, end_year = 2014)
ggplot()+
  geom_line(data = melt(result_rate$rate[c(100,100),1:90]),aes(x=Var2, y = value, group=Var1), alpha=0.2)+
  geom_line(aes(1:90,result_rate$input[1:90,ncol(result_rate$input)]), color ="red")

result_rate <- simulate_demography("death","Female",simul_time=600,n_components = 5,lambda = 5, "mean", start_year = 1960, end_year = 2014)
ggplot()+
  geom_line(data = melt(result_rate$rate[c(100,100),1:90]),aes(x=Var2, y = value, group=Var1), alpha=0.2)+
  geom_line(aes(1:90,result_rate$input[1:90,ncol(result_rate$input)]), color ="red")

result_rate <- simulate_demography("migration","Female",simul_time=600,n_components = 5,lambda = 0, "mean", start_year = 1960, end_year = 2014)
ggplot()+
  geom_line(data = melt(result_rate$rate[c(100,100),1:60]),aes(x=Var2, y = value, group=Var1), alpha=0.2)#+
  # geom_line(aes(1:60,result_rate$input[1:60,ncol(result_rate$input)]), color ="red")

result_rate <- simulate_demography("migration","Male",simul_time=600,n_components = 5,lambda = 0, "mean", start_year = 1960, end_year = 2014)
ggplot()+
  geom_line(data = melt(result_rate$rate[c(100,100),1:60]),aes(x=Var2, y = value, group=Var1), alpha=0.2)#+
  # geom_line(aes(1:60,result_rate$input[1:60,ncol(result_rate$input)]), color ="red")


sum(rev(demographics_data$N_steady)*(((result_rate[200,]+2)/2)^2)[-c(1:15)])/sum(demographics_data$N_steady)

res <- get.demographics(96,500,eps = c(rep(1, 50), rep(0, 46)), use_migration=FALSE)

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
# to Python
res <- get.demographics(96,500,eps = c(rep(1, 50), rep(0, 46)), FALSE)


Fertility =as.data.frame(res$Fertility))
save(Fertility, file="Fertility.Rda")
Mortality_male =as.data.frame(res$Mortality_male))
save(Mortality_male, file="Mortality_male.Rda")
Mortality_female =as.data.frame(res$Mortality_female))
save(Mortality_female, file="Mortality_female.Rda")


N_female_nm =as.data.frame(res$N_female)
save(N_female_nm, file="N_female_non_migration.Rda")
N_male_nm =as.data.frame(res$N_male)
save(N_male_nm, file="N_male_non_migration.Rda")
plot(rowSums(N_female_nm[1:100,]+N_male_nm[1:100,]))

res <- get.demographics(96,500,eps = c(rep(1, 50), rep(0, 46)), TRUE)
N_female =as.data.frame(res$N_female)
save(N_female, file="N_female.Rda")
N_male =as.data.frame(res$N_male)
save(N_male, file="N_male.Rda")
plot(rowSums(N_female[1:100,]+N_male[1:100,]))
real_population <- data.frame(x=2:9,size=c(146.267288, 146.544710, 146.804372, 146.880432, 146.780720,	146.748590, 146.171015, 145.557576))
max_t <- 35
ggplot(data=NULL,aes(x=2:max_t))+
  geom_line(aes(y=rowSums(N_female[2:max_t,]+N_male[2:max_t,]), color="projection"))+
  geom_line(data=real_population,aes(x = x, y =size, color="real"))
plot(rowSums(N_female[2:501,]+N_male[2:501,])/rowSums(N_female[1:500,]+N_male[1:500,]))
