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
        the data and for populating the header with header information
        from the dataframe and return both dataframe'''
        
    data = pd.read_csv(filename)
    data = data[((data['Country Name'] == 'Bangladesh') |
                 (data['Country Name'] == 'Tanzania') |
                 (data['Country Name'] == 'Norway') |
                 (data['Country Name'] == 'Switzerland')) &
                ((data['Indicator Name'] == 'Population, total') |
                 (data['Indicator Name'] == 'Access to electricity (% of population)') |
                 (data['Indicator Name'] == 'CO2 emissions (kt)') |
                 (data['Indicator Name'] == 'Electric power consumption (kWh per capita)') |                         
                 (data['Indicator Name'] == 'Total greenhouse gas emissions (% change from 1990)')|                         
                 (data['Indicator Name'] == 'Electricity production from hydroelectric sources (% of total)')
                 )]
    data.reset_index(inplace=True, drop=True)
    data.drop(['Country Code', 'Indicator Code'], inplace=True, axis=1)
    # transposing the dataframe
    data_t = data.transpose()
    # populating the header with header information
    # from the dataframe
    column = data_t.iloc[0, :]
    data_t.drop('Country Name', inplace=True, axis=0)
    data_t.columns = column
    return data, data_t


def manuplated_df():
    '''we are using this function to manipulate the transposed
    # dataframe to a new form that will be useful for analysis'''
    # Creating a new dataframe
    data_n = pd.DataFrame(columns=range(8))

    # Take the years as a list from the data dataframe
    index = data_t.index
    year_array = []
    for i in index:
        year_array.append(i)
    year = year_array[1:]
    # extracting the indicator names
    col_one_list = (data['Indicator Name'].tolist())[:6]
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
    
    # populating the 'Greenhouse gas' column with values
    gh_col = data.loc[data['Indicator Name'] == 'Total greenhouse gas emissions (% change from 1990)']
    gh_col1 = (gh_col.iloc[:, 2:]).values.tolist()
    gh_col2 = np.array(gh_col1).T.tolist()
    flattened5 = [val for sublist in gh_col2 for val in sublist]
    data_n['Total greenhouse gas emissions (% change from 1990)'] = flattened5
    
    # populating the 'Electricity production from hydroelectric sources (% of total)' column with values
    hydro_col = data.loc[data['Indicator Name'] == 'Electricity production from hydroelectric sources (% of total)']
    hydro_col1 = (hydro_col.iloc[:, 2:]).values.tolist()
    hydro_col2 = np.array(hydro_col1).T.tolist()
    flattened6 = [val for sublist in hydro_col2 for val in sublist]
    data_n['Electricity production from hydroelectric sources (% of total)'] = flattened6
    
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

# we are adding the CO2 emission and population of each country
# as seperate column to the  data_countries dataframe

data_countries = pd.DataFrame(columns=range(11))
columns = ['Year', 'CO2_S', 'CO2_B',
           'CO2_T', 'CO2_N', 'Mean_CO2(in kt)', 'pop_S',
           'pop_N', 'pop_T', 'pop_B', 'Mean_pop(in million)']
data_countries.columns = columns
data_countries['Year'] = data_T.loc[:, 'Year']
data_countries['pop_S'] = data_S['Population, total'].values.tolist()
data_countries['pop_T'] = data_T['Population, total'].values.tolist()
data_countries['pop_N'] = data_N['Population, total'].values.tolist()
data_countries['pop_B'] = data_B['Population, total'].values.tolist()
data_countries['CO2_T'] = data_T['CO2 emissions (kt)'].values.tolist()
data_countries['CO2_S'] = data_S['CO2 emissions (kt)'].values.tolist()
data_countries['CO2_N'] = data_N['CO2 emissions (kt)'].values.tolist()
data_countries['CO2_B'] = data_B['CO2 emissions (kt)'].values.tolist()

data_countries['Mean_CO2(in kt)'] = np.mean(data_countries[['CO2_T', 'CO2_S',
                                                            'CO2_N', 'CO2_B']], axis=1)
data_countries['Mean_pop(in million)'] = np.mean(data_countries[['pop_T', 'pop_S',
                                                                 'pop_N', 'pop_B']], axis=1)
# the mean value is converted to million
data_countries['Mean_pop(in million)'] = data_countries['Mean_pop(in million)']/10e+05
# we are taking 10 years value by slicing
data_countries = data_countries.iloc[50:61, :]
data_countries = data_countries.dropna()
# In a line plot over ten years periods, 
# the average CO2 emissions and populations of four countries are displayed.
# create figure and axis objects with subplots()
fig, ax = plt.subplots()
# make a plot
ax.plot(data_countries['Year'],
        data_countries['Mean_CO2(in kt)'],
        color="red",
        label='Mean of CO2 emission')
# set x-axis label
ax.set_xlabel("year", fontsize = 14)
# set y-axis label
ax.set_ylabel("Mean of 4 countries CO2 emission (kt)",
              color="red",
              fontsize=11)
# twin object for two different y-axis on the sample plot
ax2 = ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(data_countries['Year'],
         data_countries['Mean_pop(in million)'],
         color="blue",
         label='Mean of population')
ax2.set_ylabel("Mean of 4 countries population(in million)",
               color="blue",
               fontsize=10.5)
fig.suptitle("Four countries average CO2 emissions and populations over a ten-year period",
             fontsize=11)
plt.gcf().autofmt_xdate()
fig.legend([ax, ax2],     # The line objects
           labels=['Mean of CO2 emission', 'Mean of population'],   # The labels for each line
           loc="center left",   # Position of legend
           bbox_to_anchor=(0.3, 0.8)  # adjusting the position
           )
plt.show()


# inorder to calculate skewness of
# Bangladesh over a time period of 10 years
skewness = skew(data_countries['CO2_B'])

# plottting heatmap to show the relation
# between indicators of Norway and Bangladesh
# create figure and axis objects with subplots()
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 7))
# we are removing Access to electricity (% of population) column from the data
data_N.drop(['Access to electricity (% of population)',
             'Electricity production from hydroelectric sources (% of total)'],
            inplace=True,
            axis=1)
data_B.drop(['Access to electricity (% of population)',
             'Electricity production from hydroelectric sources (% of total)'],
            inplace=True,
            axis=1)
# we are plotting one heatmap in the ax1 axis and the other in ax2
res = sns.heatmap(data_N.corr(),
                  cmap='RdBu',
                  ax=ax1,
                  annot=True,
                  annot_kws={'size': 20},
                  cbar=False)
res2 = sns.heatmap(data_B.corr(),
                   cmap='RdBu',
                   ax=ax2,
                   annot=True,
                   annot_kws={'size': 20},
                   cbar=False)
res.set_xticklabels(res.get_xmajorticklabels(), fontsize=20)
res.set_yticklabels(res.get_xmajorticklabels(), fontsize=20)
res2.set_xticklabels(res.get_xmajorticklabels(), fontsize=20)
# setting the label and title
ax1.yaxis.set_label_position("left")
ax1.set_title('Norway', fontsize=25)
ax2.set_title('Bangladesh', fontsize=25)
ax2.set_yticks([])
fig.subplots_adjust(wspace=0.1)
fig.suptitle(" Heatmap for checking the correlation between the indicators",
             fontsize=32, y=1.0)
plt.show()

# plotting barplot to show the electricity access of 10 years
# for Tanzania and Switzerland
# we are taking 10 years data using slicing
data3 = data_T.iloc[50:61, :]
data4 = data_S.iloc[50:61, :]
plt.figure(figsize=(16, 18))
x = np.arange(11)
ax1 = plt.subplot(1, 1, 1)
w = 0.3
# plt.xticks(), will label the bars on x axis with the respective years.
plt.xticks(x + w / 2,
           data3['Year'].astype(int),
           rotation='vertical',
           fontsize=25)
plt.yticks(fontsize=25)
plt.ylabel('Access to electricity (% of population)', fontsize=23)
tanzania = ax1.bar(x,
                   data4['Access to electricity (% of population)'],
                   width=w,
                   color='g',
                   align='center')
# we are using two different axes that share the same x axis,
# we have used ax1.twinx() method.
ax2 = ax1.twinx()
switz = ax2.bar(x + w, data3['Access to electricity (% of population)'],
                width=w, color='b', align='center')
# Set the Y axis label as Access to electricity (% of population).

plt.yticks([])
plt.title("Data on energy access over the past 10 years for Tanzania and Switzerland",
          size=26,
          y=1)
# To set the legend on the plot we have used plt.legend()
plt.legend([tanzania, switz],
           ['elect_access_switzerland', 'elect_access_Tanzania'],
           bbox_to_anchor=(0.95, 1.005),
           ncol=2,
           fontsize=20)
# To show the plot finally we have used plt.show().
plt.show()
# plt.show()

# we are plotting a line plot to represent
# the CO2 emission and hydroelectric source of Switzerland

data3 = data_T.iloc[50:56, :]
data4 = data_S.iloc[50:56, :]

fig, ax = plt.subplots(figsize=(8, 8))
# make a plot
ax.plot(data4['Year'],
        data4['CO2 emissions (kt)'],
        color="red", 
        label='CO2 emission (kt)')
# set x-axis label
ax.set_xlabel("year", fontsize=14)
# set y-axis label
ax.set_ylabel("CO2 emission (kt)",
              color="red",
              fontsize=12)
# twin object for two different y-axis on the sample plot
ax2 = ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(data4['Year'],
         data4[( 'Electricity production from hydroelectric sources (% of total)')],
         color="blue",
         label='Electricity production from hydroelectric sources (% of total)')
ax2.set_ylabel("Electricity production from hydroelectric sources(% of total) ",
               color="blue",
               fontsize=12)
fig.suptitle("Switzerland's CO2 emissions and hydropower output",
             fontsize=17, y=0.92)
plt.gcf().autofmt_xdate()
fig.legend([ax, ax2],     # The line objects
           labels=['CO2 emission (kt)',
                   'Electricity production from hydroelectric sources'],   # The labels for each line
           loc="center left",   # Position of legend
           bbox_to_anchor=(0.2, 0.8)  # adjusting the position
           )
plt.show()