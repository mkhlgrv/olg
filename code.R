out1 <- master.function(simul_time = 500,
                        k_initial = 8,
                        K_steady_0 = 20000,
                        eps = c(rep(1, 50), rep(0, 46)),
                        tau_retirement = 0.5,
                        # retirement coefficient
                        eps_ret = c(rep(0, 50), rep(1, 46)),
                        save.demographics = TRUE, 
                        save_path_steady = TRUE,
                        G = 96,
                        tol_steady = 1/10^6,
                        update_demographics = FALSE)
out2 <- master.function(simul_time = 500,
                        k_initial = 8,
                        K_steady_0 = 20000,
                        eps = c(rep(1, 50), rep(0, 46)),
                        tau_retirement = 0.3,
                        # retirement coefficient
                        eps_ret = c(rep(0, 50), rep(1, 46)),
                        save.demographics = TRUE, 
                        save_path_steady = TRUE,
                        G = 96,
                        tol_steady = 1/10^6,
                        update_demographics = FALSE)
out3 <- master.function(simul_time = 500,
                        k_initial = 8,
                        K_steady_0 = 20000,
                        eps = c(rep(1, 50), rep(0, 46)),
                        tau_retirement = 0.2,
                        # retirement coefficient
                        eps_ret = c(rep(0, 50), rep(1, 46)),
                        save.demographics = TRUE, 
                        save_path_steady = TRUE,
                        G = 96,
                        tol_steady = 1/10^6,
                        update_demographics = FALSE)
out4 <- master.function(simul_time = 500,
                        k_initial = 8,
                        K_steady_0 = 20000,
                        eps = c(rep(1, 50), rep(0, 46)),
                        tau_retirement = 0.1,
                        # retirement coefficient
                        eps_ret = c(rep(0, 50), rep(1, 46)),
                        save.demographics = TRUE, 
                        save_path_steady = TRUE,
                        G = 96,
                        tol_steady = 1/10^6,
                        update_demographics = FALSE)



df <- as.data.frame(out1$k_history %>% t) %>%
  tibble::rownames_to_column('year') %>%
  melt %>% mutate(year = as.numeric(year)) 

Time <- 250
cons1 <- (((out1$Cons[1:Time,1:(Time+95)]*
     out1$demographics$N[1:Time,1:(Time+95)]) %>% rowSums())/
  (out1$demographics$N[1:Time,1:(Time+95)] %>% rowSums()))
cons2 <- (((out2$Cons[1:Time,1:(Time+95)]*
              out2$demographics$N[1:Time,1:(Time+95)]) %>% rowSums())/
            (out2$demographics$N[1:Time,1:(Time+95)] %>% rowSums()))
cons3 <- (((out3$Cons[1:Time,1:(Time+95)]*
              out3$demographics$N[1:Time,1:(Time+95)]) %>% rowSums())/
            (out3$demographics$N[1:Time,1:(Time+95)] %>% rowSums()))
cons4 <- (((out4$Cons[1:Time,1:(Time+95)]*
              out4$demographics$N[1:Time,1:(Time+95)]) %>% rowSums())/
            (out4$demographics$N[1:Time,1:(Time+95)] %>% rowSums()))
ggplot()+
  # geom_line(aes(1:Time, cons1, color='0.5'))+
  # geom_line(aes(1:Time, cons2, color='0.3'))+
  geom_line(aes(1:Time, cons3, color='0.2'))+
  geom_line(aes(1:Time, cons4, color='0.1'))


ggplot()+
  # geom_line(aes(1:Time, out1$k[1:Time], color='0.5'))+
  # geom_line(aes(1:Time, out2$k[1:Time], color='0.3'))+
  geom_line(aes(1:Time, out3$k[1:Time], color='0.2'))+
  geom_line(aes(1:Time, out4$k[1:Time], color='0.1'))
ggplot()+
  geom_line(data = df %>% filter(year<=600), aes(x = year, y = value, group = variable))+
  geom_abline(slope = 0, intercept = out1$k %>% last, color = 'red')

Cons <- out1$Cons
A <- out1$A
plot(diag(Cons)[1:(200)], type='l')
plot(Cons[1:96,96], type='l')
plot(A[1:50,50] %>% diff, type='l')
plot(A[400:495,495], type='l')
plot(diag(Cons[,-1])[1:(300)], type='l')
plot(r, type='l')

labor_share <- ((out1[[6]] %>% .$L)/(out1[[6]] %>% .$N %>% rowSums))
popul <- out1[[6]]