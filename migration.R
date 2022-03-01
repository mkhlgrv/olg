death_rate = read.csv('death_rate.txt', skip = 2) %>%
  mutate(Age = as.character(Age),
                                                Age = ifelse(Age == '110+', '110', Age),
                                                Age= as.integer(Age))
birth_rate = read.csv('birth_rate.txt', skip = 2) %>%
  mutate(Age = as.character(Age),
         Age = ifelse(Age == '110+', '110', Age),
         Age= as.integer(Age))
pop_size <- read.csv('pop_size.txt', skip = 2) %>% 
  mutate(Age = as.character(Age),
         Age = ifelse(Age == '110+', '110', Age),
         Age = as.numeric(Age),
         Year = as.numeric(substring(Year,1,4)))
pop_size_corrected <- pop_size %>%
  group_by(Age, Year) %>%
  summarise(Female = mean(Female), Male = mean(Male), Total = mean(Total))
pop_size_corrected %>% write.csv('pop_size_corrected.txt')

Migration <- data.frame()
for(y in 1959:2014){
  total_birth <- pop_size_corrected[pop_size_corrected$Year==y,c("Age", "Female")]  %>% na.omit()%>%
    inner_join(birth_rate[birth_rate$Year==y,], by = c("Age")) %>%
    mutate(birth=Female*ASFR) %>%
    .["birth"] %>%
    sum()
  female_male <- pop_size_corrected[pop_size_corrected$Year==y,c("Female", "Male")] %>% .[1,]
  female_ratio <- female_male[1]/(female_male[1]+female_male[2])
  
  prev_year_population <- pop_size_corrected[pop_size_corrected$Year==y,c("Age", "Female", "Male")]  %>% na.omit() %>%
    inner_join(death_rate[death_rate$Year==y,c("Age", "Female", "Male")],
               by = c("Age"), suffix=c("", "_death")) %>%
    mutate(Female = Female*(1-Female_death),
           Male = Male*(1-Male_death)) %>%
    ungroup %>%
    filter(Age!=max(Age)) %>%
    .[,c("Female", "Male")]
  
  domestic_population <- rbind(as.numeric(c(total_birth*female_ratio,total_birth*(1-female_ratio))),
                               prev_year_population)
  
  next_year_domestic <- cbind(pop_size_corrected[pop_size_corrected$Year==y,c("Female", "Male")]  %>% na.omit(),
                              domestic_population)
  ages <- unique(pop_size_corrected$Age) %>% na.omit()
  migration_ratio <- cbind(y+1,ages,(next_year_domestic[,1]-next_year_domestic[,3])/next_year_domestic[,1],
                           (next_year_domestic[,2]-next_year_domestic[,4])/next_year_domestic[,2],
                           (next_year_domestic[,1]+next_year_domestic[,2]-next_year_domestic[,3]-next_year_domestic[,4])/(
                             next_year_domestic[,1]+next_year_domestic[,2] 
                           ))
  # migration_ratio[nrow(migration_ratio),c(3,4,5)] <- rep(0,3)
  colnames(migration_ratio) <- c("Year","Age", "Female", "Male", "Total")
  Migration <- rbind(Migration,data.frame(migration_ratio))
  
}


Migration[is.infinite(Migration$Male),"Male"] <- NA
Migration[is.infinite(Migration$Female),"Female"] <- NA
Migration[is.infinite(Migration$Total),"Total"] <- NA
Migration %>% write.csv('migration_rate.txt')
plot(Migration[Migration$Year==1990,"Total"])

