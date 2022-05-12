# death_rate = read.csv('death_rate.txt') %>%
#   mutate(Age = as.character(Age),
#                                                 Age = ifelse(Age == '110+', '110', Age),
#                                                 Age= as.integer(Age))
# death_rate %>% write.csv('death_rate_corrected.txt')
# birth_rate = read.csv('birth_rate.txt') %>%
#   mutate(Age = as.character(Age),
#          Age = ifelse(Age == '12-', '12', ifelse(Age=='55+', 55, Age)),
#          Age= as.integer(Age))
# birth_rate %>% write.csv('birth_rate_corrected.txt')
# pop_size <- read.csv('pop_size.txt') %>%
#   mutate(Age = as.character(Age),
#          Age = ifelse(Age == '110+', '110', Age),
#          Age = as.numeric(Age),
#          Year = as.numeric(substring(Year,1,4)))
# pop_size_corrected <- pop_size %>%
#   group_by(Age, Year) %>%
#   summarise(Female = mean(Female), Male = mean(Male), Total = mean(Total))
# pop_size_corrected %>% write.csv('pop_size_corrected.txt')

birth_rate <- read.csv('birth_rate_corrected.txt')
death_rate <- read.csv('death_rate_corrected.txt')
pop_size <-read.csv('pop_size_corrected.txt')

get_rates_at_year_begin <- function(year, age, sex){
  death_year_age <- death_rate[death_rate$Year== year & death_rate$Age==age, sex]*
    0.5*( pop_size[pop_size$Year== year & pop_size$Age==age, sex]+
            pop_size[pop_size$Year== (year+1) & pop_size$Age==age, sex])
  death_year_age_plus1 <- death_rate[death_rate$Year== year & death_rate$Age==(age+1), sex]*
    0.5*( pop_size[pop_size$Year== year & pop_size$Age==(age+1), sex]+
            pop_size[pop_size$Year== (year+1) & pop_size$Age==(age+1), sex])
  Death <- (death_year_age+death_year_age_plus1)/2
  death_rate_at_year_begin <- Death/pop_size[pop_size$Year== year & pop_size$Age==age,sex]
  if (sex=="Female"&age>=12 & age<=54){
    birth_year_age <- birth_rate[(birth_rate$Year== year) & (birth_rate$Age==age), 'ASFR']*
      0.5*( pop_size[pop_size$Year== year & pop_size$Age==age, sex]+
              pop_size[pop_size$Year== (year+1) & pop_size$Age==age, sex])
    birth_year_age_plus1 <- birth_rate[birth_rate$Year== year & birth_rate$Age==(age+1), 'ASFR']*
      0.5*( pop_size[pop_size$Year== year & pop_size$Age==(age+1), sex]+
              pop_size[pop_size$Year== (year+1) & pop_size$Age==(age+1), sex])
    Birth <- (birth_year_age+birth_year_age_plus1)/2
    fertility_rate_at_year_begin <- Birth/pop_size[pop_size$Year== year & pop_size$Age==age,sex]
  }else{
    fertility_rate_at_year_begin <- 0
  }
  
  migrant_rate_at_year_begin <- (pop_size[pop_size$Year== (year+1) & pop_size$Age==(age+1),sex] -
                                   (pop_size[pop_size$Year== year & pop_size$Age==age,sex]
                                    - Death))/pop_size[pop_size$Year== year & pop_size$Age==age,sex]
  return(list(death_rate_at_year_begin,fertility_rate_at_year_begin,migrant_rate_at_year_begin))
  
}

Birth_rate_result <- birth_rate
Death_rate_result <- death_rate
Migration_rate_result <- death_rate
for (year in 1959:2013){
  for (age in 0:109){
    for (sex in c("Male", "Female")){
      rates <- get_rates_at_year_begin(year, age, sex)
      # print(year)
      # print(age)
      # print(sex)
      Death_rate_result[Death_rate_result$Year==year &Death_rate_result$Age==age
                        , sex] <- rates[1]
      Birth_rate_result[Birth_rate_result$Year==year &Birth_rate_result$Age==age
                          , 'ASFR'] <- rates[2]
      Migration_rate_result[Migration_rate_result$Year==year &Migration_rate_result$Age==age
                        , sex] <- rates[3]
    }
  }
}

Birth_rate_result %>% write.csv('birth_rate.txt')
Death_rate_result %>% write.csv('death_rate.txt')
Migration_rate_result %>% write.csv('migration_rate.txt')

