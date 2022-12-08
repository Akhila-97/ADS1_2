# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:50:50 2022

@author: akhil
"""
# make the necessary imports
import pandas as pd
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns


def readfile(filename):
    '''use this function to read the file and transpose 
        the data and return both dataframe'''
    data = pd.read_csv(filename)
    data = data[((data['Country Name'] == 'Bangladesh') |
                 (data['Country Name'] == 'Tanzania') |
                 (data['Country Name'] == 'Norway') |
                 (data['Country Name'] == 'Switzerland')) &
                ((data['Indicator Name'] == ' Population, total') |
                 (data['Indicator Name'] == 'Access to electricity (% of population)') |
                 (data['Indicator Name'] == 'CO2 emissions (kt)') |
                 (data['Indicator Name'] == 'Electric power consumption (kWh per capita)') 
                 )]
    
    data.reset_index(inplace=True,drop=True)
    data.drop(['Country Code', 'Indicator Code'], inplace=True, axis=1)
    data_t = data.transpose()
    column = data_t.iloc[0, :]
    data_t.drop('Country Name', inplace=True, axis=0) 
    data_t.columns = column
    return data, data_t


def manuplated_df():
    '''we are using this function to manipulate the transposed
    # dataframe to a new form that will be useful for analysis'''
    # Creating a new dataframe
    data_n = pd.DataFrame(columns=range(6))

    # Take the years as a list from the data dataframe
    index = data_t.index
    year_array = []
    for i in index:
        year_array.append(i)
    year = year_array[1:]

    # extracting the indicator names
    col_one_list = (data['Indicator Name'].tolist())[:4]
    # extracting country name using groupBy and adding as a list
    country_values = list((data.groupby('Country Name', sort=False)).groups)
    # extracting the count of countries
    countries_count = len(country_values)

    # we are adding the required coulumn names
    # along with the country_values list and reversing the complete list
    col_one_list.extend(['Country Name', 'Year'])
    col_one_list.reverse()

    # setting the column names
    data_n.columns = col_one_list

    # Loading the values of years to year column ,and repeating the year as
    # much as time a country is repeated
    data_n['Year'] = pd.Series(year).repeat(countries_count)

    # counting the length of a new dataframe
    rowcount = len(data_n.index)
    data_n.reset_index(inplace=True, drop=True)

    # calculating the number of times countries  need to be repeated
    valuecount = int(rowcount/countries_count)
    data_n['Country Name'] = country_values * (valuecount)

    # populating the 'Population ,total' column with values
    population_col = data.loc[data['Indicator Name'] == 'Population, total']
    population_col1 = (population_col.iloc[:, 2:]).values.tolist()
    population_col2 = np.array(population_col1).T.tolist()
    flattened = [val for sublist in population_col2 for val in sublist]
    data_n['Population, total'] = flattened

    # populating the 'Electric power consumption (kWh per capita)' column with values
    power_consumption_col = data.loc[data['Indicator Name'] == 'Electric power consumption (kWh per capita)']
    power_consumption1_col = (power_consumption_col.iloc[:, 2:]).values.tolist()
    power_consumption2_col = np.array(power_consumption1_col).T.tolist()
    flattened2 = [val for sublist in power_consumption2_col for val in sublist]
    data_n['Electric power consumption (kWh per capita)'] = flattened2

    # populating the 'Access to electricity (% of population)' column with values
    access_to_electricity_col = data.loc[data['Indicator Name'] == 'Access to electricity (% of population)']
    access_to_electricity_col1 = (access_to_electricity_col.iloc[:, 2:]).values.tolist()
    access_to_electricity_col2 = np.array(access_to_electricity_col1).T.tolist()
    flattened3 = [val for sublist in access_to_electricity_col2 for val in sublist]
    data_n['Access to electricity (% of population)'] = flattened3

    # populating the 'CO2 emissions (kt)' column with values
    co2_col = data.loc[data['Indicator Name'] == 'CO2 emissions (kt)']
    co2_col1 = (co2_col.iloc[:, 2:]).values.tolist()
    co2_col2 = np.array(co2_col1).T.tolist()
    flattened4 = [val for sublist in co2_col2 for val in sublist]
    data_n['CO2 emissions (kt)'] = flattened4
    return data_n


# call the function
data, data_t = readfile('ads2_data.csv')

# we are manipulating the dataframe for ease of usage
data_n = manuplated_df()

# we are seperating the data of each selected country for ease of usage
data_N = data_n.loc[data_n['Country Name'] == 'Norway']
data_B = data_n.loc[data_n['Country Name'] == 'Bangladesh']
data_T = data_n.loc[data_n['Country Name'] == 'Tanzania']
data_S = data_n.loc[data_n['Country Name'] == 'Switzerland']

# we are calculating the mean of CO2 emissions of countries
# over a time period of 10 years

# we are adding the CO2 emission of each country
# as seperate column to the  data_countries dataframe

data_countries = pd.DataFrame(columns=range(6))
columns = ['Year', 'CO2_S', 'CO2_B',
           'CO2_T', 'CO2_N', 'Mean (in kt)']
data_countries.columns = columns
data_countries['Year'] = data_T.loc[:, 'Year']
data_countries['CO2_T'] = data_T['CO2 emissions (kt)'].values.tolist()
data_countries['CO2_S'] = data_S['CO2 emissions (kt)'].values.tolist()
data_countries['CO2_N'] = data_N['CO2 emissions (kt)'].values.tolist()
data_countries['CO2_B'] = data_B['CO2 emissions (kt)'].values.tolist()
data_countries['Mean (in kt)'] = np.mean(data_countries, axis=1)

# data_countries['Mean (in million)'] = data_countries['Mean (in million)']/10e+05
# we are taking 10 years value by slicing
data_countries = data_countries.iloc[50:61, :]
data_countries = data_countries.dropna()
plt.plot(data_countries['Year'], data_countries['Mean (in kt)'])
# Label for x-axis
plt.xlabel("Year")
# Label for y-axis
plt.ylabel("Mean of 4 countries CO2 emission (kt)")
plt.title("Average CO2 emission of 4 countries over a period of 10 years")
plt.gcf().autofmt_xdate()


# inorder to calculate skewness of
# Bangladesh over a time period of 10 years
skewness = skew(data_countries['CO2_B'])

# creating a heatmap for Norway and Bangladesh inorder to
# check the correlation between various indicators
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 5))
data_N.drop(['Access to electricity (% of population)'], inplace=True, axis=1) 
sns.heatmap(data_N.corr(), cmap='RdBu', ax=ax1, annot=True, cbar=False)
data_B.drop(['Access to electricity (% of population)'], inplace=True, axis=1)
sns.heatmap(data_B.corr(), cmap='RdBu', ax=ax2, annot=True, cbar=False)
ax1.yaxis.set_label_position("left")
ax1.title.set_text('Norway')
ax2.title.set_text('Bangladesh')
ax2.set_yticks([])
fig.subplots_adjust(wspace=0.1)
fig.suptitle(" Heatmap for checking the correlation between the indicators",
             fontsize=24, y=1.01)
plt.show()

# plotting barplot to show the electricity access of 10 years
# for Bangladesh and Switzerland

# we are taking 10 years data using slicing
data3 = data_T.iloc[50:61, :]
data4 = data_S.iloc[50:61, :]
plt.figure(figsize=(16, 13))
x = np.arange(11)
ax1 = plt.subplot(1, 1, 1)
w = 0.3
# plt.xticks(), will label the bars on x axis with the respective years.
plt.xticks(x + w / 2,  data3['Year'].astype(int), rotation='vertical')
tanzania = ax1.bar(x, data3['Access to electricity (% of population)'], width=w,
                   color='b', align='center')
# we are using two different axes that share the same x axis,
# we have used ax1.twinx() method.
ax2 = ax1.twinx()
switz = ax2.bar(x + w, data4['Access to electricity (% of population)'], 
                width=w, color='g', align='center')
# Set the Y axis label as Access to electricity (% of population).
plt.ylabel('Access to electricity (% of population)')
plt.title("10 years data of electricity accesss for Bangladesh and Switzerland ", size=24)
# To set the legend on the plot we have used plt.legend()
plt.legend([tanzania, switz], ['elect_access_bngldsh', 'elect_access_switz'])
# To show the plot finally we have used plt.show().
plt.show()
