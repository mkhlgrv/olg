pop_size_female <- read.csv('pop_size_corrected.txt')  %>%
  filter(Age==0) %>% .[,"Female"]
pop_size_male <- read.csv('pop_size_corrected.txt')  %>%
  filter(Age==0) %>% .[,"Male"]
plot(pop_size_female/pop_size_male)

mean((pop_size_female/pop_size_male)[30:57])
# female_to_male_ratio=0.9489044
