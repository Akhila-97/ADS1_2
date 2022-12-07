# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:50:50 2022

@author: akhil
"""
# make the necessary imports
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


def readfile (filename) :
    '''use this function to read the file and transpose the data and return both dataframe'''
    data = pd.read_csv(filename)
    data =data[((data['Country Name'] == 'Bangladesh') |
            (data['Country Name'] == 'China') |
            (data['Country Name'] == 'Norway') |
            (data['Country Name'] == 'Switzerland')) &
            ((data['Indicator Name']=='Population, total')|
             (data['Indicator Name']=='Access to electricity (% of population)') |
             (data['Indicator Name']=='Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)') |
             (data['Indicator Name']=='Electric power consumption (kWh per capita)') 
             )]
    
    data.reset_index(inplace=True,drop=True)
    data.drop(['Country Code','Indicator Code'], inplace=True, axis=1)
    data_t = data.transpose()
    return data,data_t

# call the function 
data , data_t = readfile('ads2_data.csv')

# Creating a new dataframe
data_n = pd.DataFrame(columns = range(6))

# Take the years as a list from the data dataframe
index = data_t.index
year_array=[]
for i in index:
    year_array.append(i) 
year = year_array[2:]

# extracting the indicator names
col_one_list= (data['Indicator Name'].tolist())[:4]
# extracting country name using groupBy
country_values = list((data.groupby('Country Name',sort = False)).groups)
# extracting the count of countries
countries_count = len(country_values)

col_one_list.extend(['Country Name','Year'])
col_one_list.reverse()

# setting the column names
data_n.columns = col_one_list

# Loading the values of years to year column
data_n['Year'] = pd.Series(year).repeat(countries_count)

# counting the length of a new dataframe
rowcount = len(data_n.index)
data_n.reset_index(inplace=True,drop=True)

# calculating the number of times countries  need to be repeated
valuecount = int( rowcount/countries_count )
data_n['Country Name'] = country_values * (valuecount)

# populating the 'Population ,total' column with values 
population_col = data.loc[data['Indicator Name'] == 'Population, total' ]
population_col1 = (population_col.iloc[:,2:]).values.tolist()
population_col2 = np.array(population_col1).T.tolist()
flattened = [val for sublist in population_col2 for val in sublist]
data_n['Population, total'] = flattened

# populating the 'Electric power consumption (kWh per capita)' column with values 
power_consumption_col = data.loc[data['Indicator Name'] == 'Electric power consumption (kWh per capita)' ]
power_consumption1_col = (power_consumption_col.iloc[:,2:]).values.tolist()
power_consumption2_col = np.array(power_consumption1_col).T.tolist()
flattened2 = [val for sublist in power_consumption2_col for val in sublist]
data_n['Electric power consumption (kWh per capita)'] = flattened2

# populating the 'Access to electricity (% of population)' column with values
access_to_electricity_col = data.loc[data['Indicator Name'] == 'Access to electricity (% of population)' ]
access_to_electricity_col1 = (access_to_electricity_col.iloc[:,2:]).values.tolist()
access_to_electricity_col2 = np.array(access_to_electricity_col1).T.tolist()
flattened3 = [val for sublist in access_to_electricity_col2 for val in sublist]
data_n['Access to electricity (% of population)'] = flattened3

# populating the 'Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)' column with values
poverty_col= data.loc[data['Indicator Name'] == 'Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)' ]
poverty_col1 = (poverty_col.iloc[:,2:]).values.tolist()
poverty_col2 = np.array(poverty_col1).T.tolist()
flattened4 = [val for sublist in poverty_col2 for val in sublist]
data_n['Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)'] = flattened4

# creating a heatmap for Bangladesh inorder to 
# check the correlation between various indicators
data1 = data_n.loc[data_n['Country Name'] == 'Bangladesh' ]
data1.reset_index(inplace=True,drop=True)
data1.drop(['Access to electricity (% of population)'],inplace=True, axis=1) 
# df3.drop(['Country Name'], inplace=True, axis=1)
val =  (data1.corr())
ax = sns.heatmap(val,cmap='RdBu', vmin=-1, vmax=1, annot=True)
ax.set(xlabel="", ylabel="")

