#kmeans
################################Problem 1####################################
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pylab as plt
airlines = pd.read_excel("C:/Users/usach/Desktop/15.K-Means Clustering/EastWestAirlines (1).xlsx", sheet_name = "data")
airlines.head()
airlines.describe()
def norm_func(i) :
    x = (i - i.min())/(i.max() - i.min())
    return x
airlines_norm = norm_func(airlines.iloc[:, 1:11 ])
airlines_norm.head()
k = list(range(2,23))
sse = {}
for i in k :
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airlines_norm)
    airlines_norm["clusters"] = kmeans.labels_
    sse[i] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

model = KMeans(n_clusters = 4)
model.fit(airlines_norm)
model.labels_
md = pd.Series(model.labels_)
airlines_norm["cluster"] = md
airlines["cluster"] = md
airlines_norm.head()
clusters1 = airlines.iloc[:,[1,2,3,4,5,6,7,8,9,10]].groupby(airlines.cluster).mean()
clusters1
#cluster 3 clients has the highest usage of the airline flight, they also spend a lot in non flight transactions. this group can be classified as the highest earning group.
#Cluster 2 clients doesnt prefer to use the credit card at every ocassion, they travle the least too, this group can be classified as the least earning group.
#Cluster 1 customers do not travel a lot, however the credit card usage outside the flights transactions is the highest. This group can be classified as above the average earning group or slightly below the higher earning group.
#Cluster 0 customers are the average earning group and a good balace between expenses on trabel and non flight transactions can be seen.

#Cluster 3 customer should be give the awards to mantain these customers for a longer period.
#Cluster 1 should also be given the free flight award to encourage customer fly more on a frequent basis.
#Cluster 0 and 2 should be given discounts on flights to ensure long time enrollment with the airlines. Giving them the free flights wont be much benefical as the demand of them travelling seems to be quiet low.

#################################Problem 2#######################################
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 
# Kmeans on University Data set 
crime = pd.read_csv(r"C:/Users/usach/Desktop/15.K-Means Clustering/crime_data (1).csv")

crime.describe()
crime.columns

crime1 = crime.drop(['Unnamed: 0'], axis = 1)

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(crime1.iloc[:,:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
TWSS   
#screeplot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
from collections import Counter
Counter(mb)
crime1['clust'] = mb # creating a  new column and assigning it to new column 

crime1.head()
df_norm.head()

crime1 = crime1.iloc[:,[4,0,1,2,3]]
crime1.head()

crime1.iloc[:, 1:].groupby(crime1.clust).mean()
from collections import Counter
Counter(mb)
# Cluster 0 is the most unsafe place to staty, then comes cluster 2 and cluster 1 is the most safest place to stay in all the three. 
# For the female population too, cluster is the most unsafe and cluster 1 is the safest of all.

#############################Problem 3############################################
import pandas as pd
import seaborn
import numpy as np
import matplotlib.pyplot as plt
#loading the data frame
insurance_data = pd.read_csv("C:/Users/usach/Desktop/15.K-Means Clustering/Insurance Dataset.csv")
insurance_data.describe()
#finding the duplicate values
duplics = insurance_data.duplicated()
sum(duplics)
#checking and removing na values
insurance_data.isna().sum()
insurance_data.columns
#ploting tht boxplots
seaborn.boxplot(insurance_data["Premiums Paid"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(insurance_data["Age"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(insurance_data["Days to Renew"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(insurance_data["Claims made"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(insurance_data["Income"]);plt.title("Boxplot");plt.show()


plt.scatter(insurance_data["Premiums Paid"] , insurance_data["Age"])
plt.scatter(insurance_data["Days to Renew"] , insurance_data["Claims made"])
plt.scatter(insurance_data["Income"] , insurance_data["Premiums Paid"])
#Outlier treatment
IQR = insurance_data["Premiums Paid"].quantile(0.75) - insurance_data["Premiums Paid"].quantile(0.25)
L_limit_Premiums_Paid = insurance_data["Premiums Paid"].quantile(0.25) - (IQR * 1.5)
H_limit_Premiums_Paid = insurance_data["Premiums Paid"].quantile(0.75) + (IQR * 1.5)
insurance_data["Premiums Paid"] = pd.DataFrame(np.where(insurance_data["Premiums Paid"] > H_limit_Premiums_Paid , H_limit_Premiums_Paid ,
                                    np.where(insurance_data["Premiums Paid"] < L_limit_Premiums_Paid , L_limit_Premiums_Paid , insurance_data["Premiums Paid"])))
seaborn.boxplot(insurance_data["Premiums Paid"]);plt.title('Boxplot');plt.show()

IQR = insurance_data["Claims made"].quantile(0.75) - insurance_data["Claims made"].quantile(0.25)
L_limit_Claims_made = insurance_data["Claims made"].quantile(0.25) - (IQR * 1.5)
H_limit_Claims_made = insurance_data["Claims made"].quantile(0.75) + (IQR * 1.5)
insurance_data["Claims made"] = pd.DataFrame(np.where(insurance_data["Claims made"] > H_limit_Claims_made , H_limit_Claims_made ,
                                    np.where(insurance_data["Claims made"] < L_limit_Claims_made , L_limit_Claims_made , insurance_data["Claims made"])))
seaborn.boxplot(insurance_data["Claims made"]);plt.title('Boxplot');plt.show()

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

insurance_norm = norm_func(insurance_data)

from sklearn.cluster import KMeans
#keans model building
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(insurance_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model_insurance = KMeans(n_clusters = 3)
model_insurance.fit(insurance_norm)

model_insurance.labels_ # getting the labels of clusters assigned to each row 
cluster_insurance = pd.Series(model_insurance.labels_)  # converting numpy array into pandas series object 
insurance_data['cluster'] = cluster_insurance

insurance_data = insurance_data.iloc[:,[5,0,1,2,3,4]]
insurance_data.head()

insurance_data.iloc[:, :].groupby(insurance_data.cluster).mean()

insurance_data.to_csv("Kmeans_insurance.csv", encoding = "utf-8")

import os
os.getcwd()

################################Problem 4#############################################
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
#loading the data frame
telco_data = pd.read_excel("C:/Users/usach/Desktop/15.K-Means Clustering/Telco_customer_churn (1).xlsx")
telco_data.drop(['Count' , 'Quarter'] , axis=1 , inplace=True)
telco_data.describe()
#finding the na values
telco_data.isna().sum()
#finding dulipcate values
dupis = telco_data.duplicated()
sum(dupis)

telco_data = telco_data.drop_duplicates()

new_telco_data = pd.get_dummies(telco_data)

from sklearn.preprocessing import  OneHotEncoder

OH_enc = OneHotEncoder()

new_telco_data2 = pd.DataFrame(OH_enc.fit_transform(telco_data).toarray())

from sklearn.preprocessing import  LabelEncoder
L_enc = LabelEncoder()
telco_data['Referred a Friend'] = L_enc.fit_transform(telco_data['Referred a Friend'])
telco_data['Offer'] = L_enc.fit_transform(telco_data['Offer'])
telco_data['Phone Service'] = L_enc.fit_transform(telco_data['Phone Service'])
telco_data['Multiple Lines'] = L_enc.fit_transform(telco_data['Multiple Lines'])
telco_data['Internet Service'] = L_enc.fit_transform(telco_data['Internet Service'])
telco_data['Internet Type'] = L_enc.fit_transform(telco_data['Internet Type'])
telco_data['Online Security'] = L_enc.fit_transform(telco_data['Online Security'])
telco_data['Online Backup'] = L_enc.fit_transform(telco_data['Online Backup'])
telco_data['Device Protection Plan'] = L_enc.fit_transform(telco_data['Device Protection Plan'])
telco_data['Premium Tech Support'] = L_enc.fit_transform(telco_data['Premium Tech Support'])
telco_data['Streaming TV'] = L_enc.fit_transform(telco_data['Streaming TV'])
telco_data['Streaming Movies'] = L_enc.fit_transform(telco_data['Streaming Movies'])
telco_data['Streaming Music'] = L_enc.fit_transform(telco_data['Streaming Music'])
telco_data['Unlimited Data'] = L_enc.fit_transform(telco_data['Unlimited Data'])
telco_data['Contract'] = L_enc.fit_transform(telco_data['Contract'])
telco_data['Paperless Billing'] = L_enc.fit_transform(telco_data['Paperless Billing'])
telco_data['Payment Method'] = L_enc.fit_transform(telco_data['Payment Method'])

#boxplots
seaborn.boxplot(telco_data["Tenure in Months"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Avg Monthly Long Distance Charges"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(telco_data["Avg Monthly GB Download"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(telco_data["Monthly Charge"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Charges"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(telco_data["Total Refunds"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Extra Data Charges"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Long Distance Charges"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(telco_data["Total Revenue"]);plt.title("Boxplot");plt.show()

plt.scatter(telco_data["Tenure in Months"] , telco_data["Total Extra Data Charges"])
plt.scatter(telco_data["Monthly Charge"] , telco_data["Avg Monthly Long Distance Charges"])
plt.scatter(telco_data["Total Long Distance Charges"] , telco_data["Total Revenue"])
#Outlier treatment
IQR = telco_data["Avg Monthly GB Download"].quantile(0.75) - telco_data["Avg Monthly GB Download"].quantile(0.25)
L_limit_Avg_Monthly_GB_Download = telco_data["Avg Monthly GB Download"].quantile(0.25) - (IQR * 1.5)
H_limit_Avg_Monthly_GB_Download = telco_data["Avg Monthly GB Download"].quantile(0.75) + (IQR * 1.5)
telco_data["Avg Monthly GB Download"] = pd.DataFrame(np.where(telco_data["Avg Monthly GB Download"] > H_limit_Avg_Monthly_GB_Download , H_limit_Avg_Monthly_GB_Download ,
                                    np.where(telco_data["Avg Monthly GB Download"] < L_limit_Avg_Monthly_GB_Download , L_limit_Avg_Monthly_GB_Download , telco_data["Avg Monthly GB Download"])))
seaborn.boxplot(telco_data["Avg Monthly GB Download"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Refunds"].quantile(0.75) - telco_data["Total Refunds"].quantile(0.25)
L_limit_Total_Refunds = telco_data["Total Refunds"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Refunds = telco_data["Total Refunds"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Refunds"] = pd.DataFrame(np.where(telco_data["Total Refunds"] > H_limit_Total_Refunds , H_limit_Total_Refunds ,
                                    np.where(telco_data["Total Refunds"] < L_limit_Total_Refunds , L_limit_Total_Refunds , telco_data["Total Refunds"])))
seaborn.boxplot(telco_data["Total Refunds"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Extra Data Charges"].quantile(0.75) - telco_data["Total Extra Data Charges"].quantile(0.25)
L_limit_Total_Extra_Data_Charges = telco_data["Total Extra Data Charges"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Extra_Data_Charges = telco_data["Total Extra Data Charges"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Extra Data Charges"] = pd.DataFrame(np.where(telco_data["Total Extra Data Charges"] > H_limit_Total_Extra_Data_Charges , H_limit_Total_Extra_Data_Charges ,
                                    np.where(telco_data["Total Extra Data Charges"] < L_limit_Total_Extra_Data_Charges , L_limit_Total_Extra_Data_Charges , telco_data["Total Extra Data Charges"])))
seaborn.boxplot(telco_data["Total Extra Data Charges"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Long Distance Charges"].quantile(0.75) - telco_data["Total Long Distance Charges"].quantile(0.25)
L_limit_Total_Long_Distance_Charges = telco_data["Total Long Distance Charges"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Long_Distance_Charges = telco_data["Total Long Distance Charges"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Long Distance Charges"] = pd.DataFrame(np.where(telco_data["Total Long Distance Charges"] > H_limit_Total_Long_Distance_Charges , H_limit_Total_Long_Distance_Charges ,
                                    np.where(telco_data["Total Long Distance Charges"] < L_limit_Total_Long_Distance_Charges , L_limit_Total_Long_Distance_Charges , telco_data["Total Long Distance Charges"])))
seaborn.boxplot(telco_data["Total Long Distance Charges"]);plt.title('Boxplot');plt.show()

IQR = telco_data["Total Revenue"].quantile(0.75) - telco_data["Total Revenue"].quantile(0.25)
L_limit_Total_Revenue = telco_data["Total Revenue"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Revenue = telco_data["Total Revenue"].quantile(0.75) + (IQR * 1.5)
telco_data["Total Revenue"] = pd.DataFrame(np.where(telco_data["Total Revenue"] > H_limit_Total_Revenue , H_limit_Total_Revenue ,
                                    np.where(telco_data["Total Revenue"] < L_limit_Total_Revenue , L_limit_Total_Revenue , telco_data["Total Revenue"])))
seaborn.boxplot(telco_data["Total Revenue"]);plt.title('Boxplot');plt.show()

def std_fun(i):
    x = (i-i.mean()) / (i.std())
    return (x)

telco_data_norm = std_fun(new_telco_data)

str(telco_data_norm)

from sklearn.cluster import KMeans
#kmeans model building
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(telco_data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS

plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

model_telco = KMeans(n_clusters = 3)
model_telco.fit(telco_data_norm)

model_telco.labels_ # getting the labels of clusters assigned to each row 
cluster_telco = pd.Series(model_telco.labels_)  # converting numpy array into pandas series object 
telco_data['cluster'] = cluster_telco

telco_data.head()

telco_data.iloc[:,:].groupby(telco_data.cluster).mean()

telco_data.to_csv("Kmeans_telco.csv", encoding = "utf-8")

import os
os.getcwd()


################################Problem 5################################################
import pandas as pd
import seaborn
import numpy as np
import matplotlib.pyplot as plt
#loading the dataframe
auto_data = pd.read_csv("C:/Users/usach/Desktop/15.K-Means Clustering/AutoInsurance (1).csv")
auto_data.describe()
auto_data.dtypes
auto_data.drop(['Customer'] , axis= 1 , inplace = True)

new_auto_data = auto_data.iloc[ : ,1:]
#checking for duplicates and removing duplicates if any
duplis = new_auto_data.duplicated()
sum(duplis)

new_auto_data = new_auto_data.drop_duplicates()
#checking for na and null values

new_auto_data.isna().sum()


dummy_auto_data = pd.get_dummies(new_auto_data)
#Boxplots
seaborn.boxplot(new_auto_data["Customer Lifetime Value"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_auto_data["Income"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_auto_data["Monthly Premium Auto"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_auto_data["Months Since Last Claim"]);plt.title("Boxplot");plt.show()
seaborn.boxplot(new_auto_data["Months Since Policy Inception"]);plt.title("Boxplot");plt.show()

seaborn.boxplot(new_auto_data["Total Claim Amount"]);plt.title("Boxplot");plt.show()

plt.scatter(new_auto_data["Customer Lifetime Value"] , new_auto_data["Income"])
plt.scatter(new_auto_data["Monthly Premium Autos"] , new_auto_data["Months Since Last Claime"])
plt.scatter(new_auto_data["Months Since Policy Inception"] , new_auto_data["Total Claim Amount"])
#Outlier treatment
IQR = new_auto_data["Customer Lifetime Value"].quantile(0.75) - new_auto_data["Customer Lifetime Value"].quantile(0.25)
L_limit_Customer_Lifetime_Value = new_auto_data["Customer Lifetime Value"].quantile(0.25) - (IQR * 1.5)
H_limit_Customer_Lifetime_Value = new_auto_data["Customer Lifetime Value"].quantile(0.75) + (IQR * 1.5)
new_auto_data["Customer Lifetime Value"] = pd.DataFrame(np.where(new_auto_data["Customer Lifetime Value"] > H_limit_Customer_Lifetime_Value , H_limit_Customer_Lifetime_Value ,
                                    np.where(new_auto_data["Customer Lifetime Value"] < L_limit_Customer_Lifetime_Value , L_limit_Customer_Lifetime_Value , new_auto_data["Customer Lifetime Value"])))
seaborn.boxplot(new_auto_data["Customer Lifetime Value"]);plt.title('Boxplot');plt.show()

IQR = new_auto_data["Monthly Premium Auto"].quantile(0.75) - new_auto_data["Monthly Premium Auto"].quantile(0.25)
L_limit_Monthly_Premium_Auto = new_auto_data["Monthly Premium Auto"].quantile(0.25) - (IQR * 1.5)
H_limit_Monthly_Premium_Auto = new_auto_data["Monthly Premium Auto"].quantile(0.75) + (IQR * 1.5)
new_auto_data["Monthly Premium Auto"] = pd.DataFrame(np.where(new_auto_data["Monthly Premium Auto"] > H_limit_Monthly_Premium_Auto , H_limit_Monthly_Premium_Auto ,
                                    np.where(new_auto_data["Monthly Premium Auto"] < L_limit_Monthly_Premium_Auto , L_limit_Monthly_Premium_Auto , new_auto_data["Monthly Premium Auto"])))
seaborn.boxplot(new_auto_data["Monthly Premium Auto"]);plt.title('Boxplot');plt.show()

IQR = new_auto_data["Total Claim Amount"].quantile(0.75) - new_auto_data["Total Claim Amount"].quantile(0.25)
L_limit_Total_Claim_Amount = new_auto_data["Total Claim Amount"].quantile(0.25) - (IQR * 1.5)
H_limit_Total_Claim_Amount = new_auto_data["Total Claim Amount"].quantile(0.75) + (IQR * 1.5)
new_auto_data["Total Claim Amount"] = pd.DataFrame(np.where(new_auto_data["Total Claim Amount"] > H_limit_Total_Claim_Amount , H_limit_Total_Claim_Amount ,
                                    np.where(new_auto_data["Total Claim Amount"] < L_limit_Total_Claim_Amount , L_limit_Total_Claim_Amount , new_auto_data["Total Claim Amount"])))
seaborn.boxplot(new_auto_data["Total Claim Amount"]);plt.title('Boxplot');plt.show()



def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

auto_data_norm = norm_func(dummy_auto_data)

from sklearn.cluster import KMeans
#Kmeans model building
#getting cluster values for the cluster from 2 - 9
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(auto_data_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
#elbowcurve ploting to choose best cluster value
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")
#model Building
model_auto = KMeans(n_clusters = 3)
model_auto.fit(auto_data_norm)

model_auto.labels_ # getting the labels of clusters assigned to each row 
cluster_auto = pd.Series(model_auto.labels_)  # converting numpy array into pandas series object 
auto_data['cluster'] = cluster_auto

auto_data = auto_data.iloc[:,[23,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]]
auto_data.head()

auto_data.iloc[:, :].groupby(auto_data.cluster).mean()

auto_data.to_csv("Kmeans_auto.csv", encoding = "utf-8")

import os
os.getcwd()
