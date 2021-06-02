#K-MEANS CLustering
######################Problem 1 #############################
library(data.table)
library(readxl)
EastWestAirlines <- read_excel(file.choose(), sheet = "data")
#View(EastWestAirlines)
colnames(EastWestAirlines)
ncol(EastWestAirlines)
airline_kmeans <- kmeans(norm_airline,5)
str(airline_kmeans)
airline_kmeans$centers
EastWestAirlines_New <- cbind(EastWestAirlines_New, airline_kmeans$cluster)
colnames(EastWestAirlines_New)
# Aggregate
aggregate(EastWestAirlines_New[,2:12],by= list(EastWestAirlines_New$`airline_kmeans$cluster`), FUN = mean)
# install.packages("cluster")
library(cluster)
# Using Clara function(Clustering for Large Applications) to find cluster
xcl <- clara(norm_airline,5) #Using Centroid
clusplot(xcl)
#using Partition Arround Medoids to find cluster
xpm <- pam(norm_airline,5) # Using Medoids
clusplot(xpm)

#2
crime_data <- read.csv("C:/Users/usach/Desktop/K-Means Clustering/crime_data (1).csv")
View(crime_data)
crime<-scale(crime_data[,2:5])
fitK<-kmeans(crime,4)
str(fitK)
output_final<-data.frame(crime_data,fitK$cluster)
output_final
aggregate(crime_data[,2:5],by=list(fitK$cluster),FUN=mean)
wsscluster<-(nrow(crime)-1)*sum(apply(crime,2,var))
for(i in 2:6) wsscluster[i]=sum(kmeans(crime,centers = i)$wsscluster)
plot(1:6,wsscluster,type="b",xlab="clusters",ylab = "wsscluster")
fitK$centers
fitK$cluster
#best cluster is 2
## As per above algorithm we found the maximum four cluster in crime data set & 
## also we can say that  assault crime rate is more as compare to all other crimes.   

########################Problem 2 ###########################################
install.packages("readr")
library(readr)
crime_data <- read_csv("C:/Users/usach/Desktop/K-Means Clustering/crime_data (1).csv")
new_crime_data <- crime_data[ , c(2:5)]

sum(is.na(new_crime_data))

summary(new_crime_data)

dup<- duplicated(new_crime_data)
sum(dup)
new_crime_data <- new_crime_data[!dup , ]
str(new_crime_data)

boxplot(new_crime_data$Murder)
boxplot(new_crime_data$Assault)
boxplot(new_crime_data$UrbanPop)
boxplot(new_crime_data$Rape)

qunt_Rape <- quantile(new_crime_data$Rape , probs = c(.25 , .75))
winso_Rape <- quantile(new_crime_data$Rape , probs = c(.01 , .95) , na.rm = TRUE)
H_Rape <- 1.5*IQR(new_crime_data$Rape , na.rm = TRUE)
new_crime_data$Rape[new_crime_data$Rape<(qunt_Rape[1]-H_Rape)] <- winso_Rape[1]
new_crime_data$Rape[new_crime_data$Rape>(qunt_Rape[2]+H_Rape)] <- winso_Rape[2]
boxplot(new_crime_data$Rape)


norm_crime_data <- scale(new_crime_data)
norm_crime_data <- as.data.frame(norm_crime_data)
summary(norm_crime_data)

#bivariate analysis
#scatterplot for some features of df norm_data
install.packages("ggplot2")
library(ggplot2)
qplot(Murder,Assault ,data = norm_crime_data,geom = "point")
qplot(UrbanPop,Rape,data = norm_crime_data,geom = "point")
qplot(Murder,UrbanPop,data = norm_crime_data,geom = "point")

#Model Building
# Elbow curve to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(norm_crime_data, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

# 3 Cluster Solution
fit_crime <- kmeans(norm_crime_data, 3) 
str(fit_crime)
fit_crime$cluster
final_crime <- data.frame(fit_crime$cluster, crime_data) # Append cluster membership

aggregate(crime_data[, 1:5], by = list(fit_crime$cluster), FUN = mean)

install.packages("readr")
library(readr)
write_csv(final_crime, "Kmeans_crime.csv")
getwd()
#########################Problem 3#######################################
install.packages("readr")
library(readr)

insurance_data <- read_csv("C:/Users/usach/Desktop/K-Means Clustering/Insurance Dataset.csv")


sum(is.na(insurance_data))

summary(insurance_data)

dup<- duplicated(insurance_data)
sum(dup)
insurance_data <- insurance_data[!dup , ]
str(insurance_data)

boxplot(insurance_data$`Premiums Paid`)
boxplot(insurance_data$Age)
boxplot(insurance_data$`Days to Renew`)
boxplot(insurance_data$`Claims made`)
boxplot(insurance_data$`Income`)

qunt_Premiums_Paid <- quantile(insurance_data$`Premiums Paid` , probs = c(.25 , .75))
winso_Premiums_Paid <- quantile(insurance_data$`Premiums Paid` , probs = c(.01 , .90) , na.rm = TRUE)
H_Premiums_Paid <- 1.5*IQR(insurance_data$`Premiums Paid` , na.rm = TRUE)
insurance_data$`Premiums Paid`[insurance_data$`Premiums Paid`<(qunt_Premiums_Paid[1]-H_Premiums_Paid)] <- winso_Premiums_Paid[1]
insurance_data$`Premiums Paid`[insurance_data$`Premiums Paid`>(qunt_Premiums_Paid[2]+H_Premiums_Paid)] <- winso_Premiums_Paid[2]
boxplot(insurance_data$`Premiums Paid`)

qunt_Claims_made <- quantile(insurance_data$`Claims made` , probs = c(.25 , .75))
winso_Claims_made <- quantile(insurance_data$`Claims made` , probs = c(.01 , .90) , na.rm = TRUE)
H_Claims_made <- 1.5*IQR(insurance_data$`Claims made` , na.rm = TRUE)
insurance_data$`Claims made`[insurance_data$`Claims made`<(qunt_Claims_made[1]-H_Claims_made)] <- winso_Claims_made[1]
insurance_data$`Claims made`[insurance_data$`Claims made`>(qunt_Claims_made[2]+H_Claims_made)] <- winso_Claims_made[2]
boxplot(insurance_data$`Claims made`)


install.packages("ggplot2")
library(ggplot2)
qplot(insurance_data$Income,insurance_data$"Claims made" ,data = insurance_data,geom = "point")
qplot(insurance_data$"Days to Renew",insurance_data$"Age",data = insurance_data,geom = "point")

norm_insurance_data <- scale(insurance_data)
norm_insurance_data <- as.data.frame(norm_insurance_data)
summary(norm_insurance_data)

#Model Building
# Elbow curve to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(norm_insurance_data, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

# 3 Cluster Solution
fit_insurance <- kmeans(norm_insurance_data, 3) 
str(fit_insurance)
fit_insurance$cluster
final_insurance <- data.frame(fit_insurance$cluster, insurance_data) # Append cluster membership

aggregate(insurance_data[, 1:5], by = list(fit_insurance$cluster), FUN = mean)

install.packages("readr")
library(readr)
write_csv(final_insurance, "Kmeans_insurance.csv")
getwd()

#############################Problem 4 ####################################
install.packages("readxl")
library(readxl)

telco_data <- read_excel("C:\\Users\\hp\\Desktop\\kmeans_assi\\Telco_customer_churn.xlsx")
my_telco_data <-  telco_data[ , c(-1,-2,-3)]
sum(is.na(my_telco_data))

summary(my_telco_data)

dups <- duplicated(my_telco_data)
sum(dups)
my_telco_data <- my_telco_data[!dups , ]
str(my_telco_data)

boxplot(my_telco_data["Tenure in Months"])
boxplot(my_telco_data["Avg Monthly Long Distance Charges"])
boxplot(my_telco_data["Avg Monthly GB Download"])
boxplot(my_telco_data["Monthly Charge"])
boxplot(my_telco_data["Total Charges"])
boxplot(my_telco_data["Total Refunds"])
boxplot(my_telco_data["Total Extra Data Charges"])
boxplot(my_telco_data["Total Long Distance Charges"])
boxplot(my_telco_data["Total Revenue"])

qunt_Avg_Monthly_GB_Download <- quantile(my_telco_data$"Avg Monthly GB Download" , probs = c(.25 , .75))
winso_Avg_Monthly_GB_Download <- quantile(my_telco_data$"Avg Monthly GB Download" , probs = c(.01 , .93) , na.rm = TRUE)
H_Avg_Monthly_GB_Download <- 1.5*IQR(my_telco_data$"Avg Monthly GB Download" , na.rm = TRUE)
my_telco_data$"Avg Monthly GB Download"[my_telco_data$"Avg Monthly GB Download"<(qunt_Avg_Monthly_GB_Download[1]-H_Avg_Monthly_GB_Download)] <- winso_Avg_Monthly_GB_Download[1]
my_telco_data$"Avg Monthly GB Download"[my_telco_data$"Avg Monthly GB Download">(qunt_Avg_Monthly_GB_Download[2]+H_Avg_Monthly_GB_Download)] <- winso_Avg_Monthly_GB_Download[2]
boxplot(my_telco_data$"Avg Monthly GB Download")

qunt_Total_Refunds <- quantile(my_telco_data$"Total Refunds" , probs = c(.25 , .75))
winso_Total_Refunds <- quantile(my_telco_data$"Total Refunds" , probs = c(.01 , .92) , na.rm = TRUE)
H_Total_Refunds <- 1.5*IQR(my_telco_data$"Total Refunds" , na.rm = TRUE)
my_telco_data$"Total Refunds"[my_telco_data$"Total Refunds"<(qunt_Total_Refunds[1]-H_Total_Refunds)] <- winso_Total_Refunds[1]
my_telco_data$"Total Refunds"[my_telco_data$"Total Refunds">(qunt_Total_Refunds[2]+H_Total_Refunds)] <- winso_Total_Refunds[2]
boxplot(my_telco_data$"Total Refunds")

qunt_Total_Extra_Data_Charges <- quantile(my_telco_data$"Total Extra Data Charges" , probs = c(.25 , .75))
winso_Total_Extra_Data_Charges <- quantile(my_telco_data$"Total Extra Data Charges" , probs = c(.01 , .85) , na.rm = TRUE)
H_Total_Extra_Data_Charges <- 1.5*IQR(my_telco_data$"Total Extra Data Charges" , na.rm = TRUE)
my_telco_data$"Total Extra Data Charges"[my_telco_data$"Total Extra Data Charges"<(qunt_Total_Extra_Data_Charges[1]-H_Total_Extra_Data_Charges)] <- winso_Total_Extra_Data_Charges[1]
my_telco_data$"Total Extra Data Charges"[my_telco_data$"Total Extra Data Charges">(qunt_Total_Extra_Data_Charges[2]+H_Total_Extra_Data_Charges)] <- winso_Total_Extra_Data_Charges[2]
boxplot(my_telco_data$"Total Extra Data Charges")

qunt_Total_Long_Distance_Charges <- quantile(my_telco_data$"Total Long Distance Charges" , probs = c(.25 , .75))
winso_Total_Long_Distance_Charges <- quantile(my_telco_data$"Total Long Distance Charges" , probs = c(.01 , .95) , na.rm = TRUE)
H_Total_Long_Distance_Charges <- 1.5*IQR(my_telco_data$"Total Long Distance Charges" , na.rm = TRUE)
my_telco_data$"Total Long Distance Charges"[my_telco_data$"Total Long Distance Charges"<(qunt_Total_Long_Distance_Charges[1]-H_Total_Long_Distance_Charges)] <- winso_Total_Long_Distance_Charges[1]
my_telco_data$"Total Long Distance Charges"[my_telco_data$"Total Long Distance Charges">(qunt_Total_Long_Distance_Charges[2]+H_Total_Long_Distance_Charges)] <- winso_Total_Long_Distance_Charges[2]
boxplot(my_telco_data$"Total Long Distance Charges")

qunt_Total_Revenue <- quantile(my_telco_data$"Total Revenue" , probs = c(.25 , .75))
winso_Total_Revenue <- quantile(my_telco_data$"Total Revenue" , probs = c(.01 , .99) , na.rm = TRUE)
H_Total_Revenue <- 1.5*IQR(my_telco_data$"Total Revenue" , na.rm = TRUE)
my_telco_data$"Total Revenue"[my_telco_data$"Total Revenue"<(qunt_Total_Revenue[1]-H_Total_Revenue)] <- winso_Total_Revenue[1]
my_telco_data$"Total Revenue"[my_telco_data$"Total Revenue">(qunt_Total_Revenue[2]+H_Total_Revenue)] <- winso_Total_Revenue[2]
boxplot(my_telco_data$"Total Revenue")

install.packages("ggplot2")
library(ggplot2)
qplot(my_telco_data$"Total Revenue",my_telco_data$"Total Long Distance Charges" ,data = my_telco_data,geom = "point")
qplot(my_telco_data$"Total Charges",my_telco_data$"Total Refunds",data = my_telco_data,geom = "point")


install.packages("fastDummies")
library(fastDummies)

my_telco_data_dummy <- dummy_cols(my_telco_data , remove_first_dummy = TRUE ,remove_selected_columns = TRUE)

norm_telco_data <- scale(my_telco_data_dummy)
norm_telco_data <- as.data.frame(norm_telco_data)
summary(norm_telco_data)

#Model Building
# Elbow curve to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(my_telco_data_dummy, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

# 3 Cluster Solution
fit_telco <- kmeans(my_telco_data_dummy, 3) 
str(fit_telco)
fit_telco$cluster
final_telco <- data.frame(fit_telco$cluster, telco_data) # Append cluster membership

aggregate(telco_data[, 1:30], by = list(fit_telco$cluster), FUN = mean)

install.packages("readr")
library(readr)
write_csv(final_telco, "Kmeans_telco.csv")
getwd()
########################## problem 4 #############################
install.packages("readxl")
library(readxl)

telco_data <- read_excel("C:/Users/usach/Desktop/K-Means Clustering/Telco_customer_churn (1).xlsx")
my_telco_data <-  telco_data[ , c(-1,-2,-3)]
sum(is.na(my_telco_data))

summary(my_telco_data)

dups <- duplicated(my_telco_data)
sum(dups)
my_telco_data <- my_telco_data[!dups , ]
str(my_telco_data)

boxplot(my_telco_data["Tenure in Months"])
boxplot(my_telco_data["Avg Monthly Long Distance Charges"])
boxplot(my_telco_data["Avg Monthly GB Download"])
boxplot(my_telco_data["Monthly Charge"])
boxplot(my_telco_data["Total Charges"])
boxplot(my_telco_data["Total Refunds"])
boxplot(my_telco_data["Total Extra Data Charges"])
boxplot(my_telco_data["Total Long Distance Charges"])
boxplot(my_telco_data["Total Revenue"])

qunt_Avg_Monthly_GB_Download <- quantile(my_telco_data$"Avg Monthly GB Download" , probs = c(.25 , .75))
winso_Avg_Monthly_GB_Download <- quantile(my_telco_data$"Avg Monthly GB Download" , probs = c(.01 , .93) , na.rm = TRUE)
H_Avg_Monthly_GB_Download <- 1.5*IQR(my_telco_data$"Avg Monthly GB Download" , na.rm = TRUE)
my_telco_data$"Avg Monthly GB Download"[my_telco_data$"Avg Monthly GB Download"<(qunt_Avg_Monthly_GB_Download[1]-H_Avg_Monthly_GB_Download)] <- winso_Avg_Monthly_GB_Download[1]
my_telco_data$"Avg Monthly GB Download"[my_telco_data$"Avg Monthly GB Download">(qunt_Avg_Monthly_GB_Download[2]+H_Avg_Monthly_GB_Download)] <- winso_Avg_Monthly_GB_Download[2]
boxplot(my_telco_data$"Avg Monthly GB Download")

qunt_Total_Refunds <- quantile(my_telco_data$"Total Refunds" , probs = c(.25 , .75))
winso_Total_Refunds <- quantile(my_telco_data$"Total Refunds" , probs = c(.01 , .92) , na.rm = TRUE)
H_Total_Refunds <- 1.5*IQR(my_telco_data$"Total Refunds" , na.rm = TRUE)
my_telco_data$"Total Refunds"[my_telco_data$"Total Refunds"<(qunt_Total_Refunds[1]-H_Total_Refunds)] <- winso_Total_Refunds[1]
my_telco_data$"Total Refunds"[my_telco_data$"Total Refunds">(qunt_Total_Refunds[2]+H_Total_Refunds)] <- winso_Total_Refunds[2]
boxplot(my_telco_data$"Total Refunds")

qunt_Total_Extra_Data_Charges <- quantile(my_telco_data$"Total Extra Data Charges" , probs = c(.25 , .75))
winso_Total_Extra_Data_Charges <- quantile(my_telco_data$"Total Extra Data Charges" , probs = c(.01 , .85) , na.rm = TRUE)
H_Total_Extra_Data_Charges <- 1.5*IQR(my_telco_data$"Total Extra Data Charges" , na.rm = TRUE)
my_telco_data$"Total Extra Data Charges"[my_telco_data$"Total Extra Data Charges"<(qunt_Total_Extra_Data_Charges[1]-H_Total_Extra_Data_Charges)] <- winso_Total_Extra_Data_Charges[1]
my_telco_data$"Total Extra Data Charges"[my_telco_data$"Total Extra Data Charges">(qunt_Total_Extra_Data_Charges[2]+H_Total_Extra_Data_Charges)] <- winso_Total_Extra_Data_Charges[2]
boxplot(my_telco_data$"Total Extra Data Charges")

qunt_Total_Long_Distance_Charges <- quantile(my_telco_data$"Total Long Distance Charges" , probs = c(.25 , .75))
winso_Total_Long_Distance_Charges <- quantile(my_telco_data$"Total Long Distance Charges" , probs = c(.01 , .95) , na.rm = TRUE)
H_Total_Long_Distance_Charges <- 1.5*IQR(my_telco_data$"Total Long Distance Charges" , na.rm = TRUE)
my_telco_data$"Total Long Distance Charges"[my_telco_data$"Total Long Distance Charges"<(qunt_Total_Long_Distance_Charges[1]-H_Total_Long_Distance_Charges)] <- winso_Total_Long_Distance_Charges[1]
my_telco_data$"Total Long Distance Charges"[my_telco_data$"Total Long Distance Charges">(qunt_Total_Long_Distance_Charges[2]+H_Total_Long_Distance_Charges)] <- winso_Total_Long_Distance_Charges[2]
boxplot(my_telco_data$"Total Long Distance Charges")

qunt_Total_Revenue <- quantile(my_telco_data$"Total Revenue" , probs = c(.25 , .75))
winso_Total_Revenue <- quantile(my_telco_data$"Total Revenue" , probs = c(.01 , .99) , na.rm = TRUE)
H_Total_Revenue <- 1.5*IQR(my_telco_data$"Total Revenue" , na.rm = TRUE)
my_telco_data$"Total Revenue"[my_telco_data$"Total Revenue"<(qunt_Total_Revenue[1]-H_Total_Revenue)] <- winso_Total_Revenue[1]
my_telco_data$"Total Revenue"[my_telco_data$"Total Revenue">(qunt_Total_Revenue[2]+H_Total_Revenue)] <- winso_Total_Revenue[2]
boxplot(my_telco_data$"Total Revenue")

install.packages("ggplot2")
library(ggplot2)
qplot(my_telco_data$"Total Revenue",my_telco_data$"Total Long Distance Charges" ,data = my_telco_data,geom = "point")
qplot(my_telco_data$"Total Charges",my_telco_data$"Total Refunds",data = my_telco_data,geom = "point")


install.packages("fastDummies")
library(fastDummies)

my_telco_data_dummy <- dummy_cols(my_telco_data , remove_first_dummy = TRUE ,remove_selected_columns = TRUE)

norm_telco_data <- scale(my_telco_data_dummy)
norm_telco_data <- as.data.frame(norm_telco_data)
summary(norm_telco_data)

#Model Building
# Elbow curve to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(my_telco_data_dummy, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

# 3 Cluster Solution
fit_telco <- kmeans(my_telco_data_dummy, 3) 
str(fit_telco)
fit_telco$cluster
final_telco <- data.frame(fit_telco$cluster, telco_data) # Append cluster membership

aggregate(telco_data[, 1:30], by = list(fit_telco$cluster), FUN = mean)

install.packages("readr")
library(readr)
write_csv(final_telco, "Kmeans_telco.csv")
getwd()

##############################problem 5############################
install.packages("readr")
library(readr)

auto_data <- read_csv("/C:/Users/usach/Desktop/K-Means Clustering/AutoInsurance (1).csv")
new_auto_data <- auto_data[-1]

sum(is.na(new_auto_data))

summary(new_auto_data)

dup<- duplicated(new_auto_data)
sum(dup)
new_auto_data <- new_auto_data[!dup , ]
str(new_auto_data)

boxplot(new_auto_data$`Customer Lifetime Value`)
boxplot(new_auto_data$Income)
boxplot(new_auto_data$`Monthly Premium Auto`)
boxplot(new_auto_data$`Months Since Last Claim`)
boxplot(new_auto_data$`Months Since Policy Inception`)
boxplot(new_auto_data$`Total Claim Amount`)

qunt_Customer_Lifetime_Value <- quantile(new_auto_data$`Customer Lifetime Value` , probs = c(.25 , .75))
winso_Customer_Lifetime_Value <- quantile(new_auto_data$`Customer Lifetime Value` , probs = c(.01 , .90) , na.rm = TRUE)
H_Customer_Lifetime_Value <- 1.5*IQR(new_auto_data$`Customer Lifetime Value` , na.rm = TRUE)
new_auto_data$`Customer Lifetime Value`[new_auto_data$`Customer Lifetime Value`<(qunt_Customer_Lifetime_Value[1]-H_Customer_Lifetime_Value)] <- winso_Customer_Lifetime_Value[1]
new_auto_data$`Customer Lifetime Value`[new_auto_data$`Customer Lifetime Value`>(qunt_Customer_Lifetime_Value[2]+H_Customer_Lifetime_Value)] <- winso_Customer_Lifetime_Value[2]
boxplot(new_auto_data$`Customer Lifetime Value`)

qunt_Monthly_Premium_Auto <- quantile(new_auto_data$`Monthly Premium Auto` , probs = c(.25 , .75))
winso_Monthly_Premium_Auto <- quantile(new_auto_data$`Monthly Premium Auto` , probs = c(.01 , .95) , na.rm = TRUE)
H_Monthly_Premium_Auto <- 1.5*IQR(new_auto_data$`Monthly Premium Auto` , na.rm = TRUE)
new_auto_data$`Monthly Premium Auto`[new_auto_data$`Monthly Premium Auto`<(qunt_Monthly_Premium_Auto[1]-H_Monthly_Premium_Auto)] <- winso_Monthly_Premium_Auto[1]
new_auto_data$`Monthly Premium Auto`[new_auto_data$`Monthly Premium Auto`>(qunt_Monthly_Premium_Auto[2]+H_Monthly_Premium_Auto)] <- winso_Monthly_Premium_Auto[2]
boxplot(new_auto_data$`Monthly Premium Auto`)

qunt_Total_Claim_Amount <- quantile(new_auto_data$`Total Claim Amount` , probs = c(.25 , .75))
winso_Total_Claim_Amount <- quantile(new_auto_data$`Total Claim Amount` , probs = c(.01 , .95) , na.rm = TRUE)
H_Total_Claim_Amount <- 1.5*IQR(new_auto_data$`Total Claim Amount` , na.rm = TRUE)
new_auto_data$`Total Claim Amount`[new_auto_data$`Total Claim Amount`<(qunt_Total_Claim_Amount[1]-H_Total_Claim_Amount)] <- winso_Total_Claim_Amount[1]
new_auto_data$`Total Claim Amount`[new_auto_data$`Total Claim Amount`>(qunt_Total_Claim_Amount[2]+H_Total_Claim_Amount)] <- winso_Total_Claim_Amount[2]
boxplot(new_auto_data$`Total Claim Amount`)


install.packages("ggplot2")
library(ggplot2)
qplot(new_auto_data$Income,new_auto_data$"Monthly Premium Auto" ,data = new_auto_data,geom = "point")
qplot(new_auto_data$"Total Claim Amount",new_auto_data$"Months Since Policy Inception",data = new_auto_data,geom = "point")


install.packages("fastDummies")
library(fastDummies)

my_auto_data_dummy <- dummy_cols(new_auto_data , remove_first_dummy = TRUE ,remove_selected_columns = TRUE)

norm_auto_data <- scale(my_auto_data_dummy)
summary(norm_auto_data)

#Model Building
# Elbow curve to decide the k value
twss <- NULL
for (i in 2:8) {
  twss <- c(twss, kmeans(norm_auto_data, centers = i)$tot.withinss)
}
twss

# Look for an "elbow" in the scree plot
plot(2:8, twss, type = "b", xlab = "Number of Clusters", ylab = "Within groups sum of squares")
title(sub = "K-Means Clustering Scree-Plot")

# 3 Cluster Solution
fit_auto <- kmeans(norm_auto_data, 3) 
str(fit_auto)
fit_auto$cluster
final_auto <- data.frame(fit_auto$cluster, new_auto_data) # Append cluster membership

aggregate(new_auto_data[, 1:23], by = list(fit_auto$cluster), FUN = mean)

install.packages("readr")
library(readr)
write_csv(final_auto, "Kmeans_airlines.csv")
getwd()