# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:50:50 2022

@author: akhil
"""
# make the necessary imports
import pandas as pd
import numpy as np


def readfile (filename) :
    '''use this function to read the file and transpose the data and return both dataframe'''
    data = pd.read_csv(filename)
    data =data[((data['Country Name'] == 'India') |
            (data['Country Name'] == 'China') |
            (data['Country Name'] == 'Norway') |
            (data['Country Name'] == 'Switzerland')) &
            ((data['Indicator Name']=='Population, total')|
             (data['Indicator Name']=='Electricity production from coal sources (% of total)') |
             (data['Indicator Name']=='Electric power consumption (kWh per capita)') 
             )]
    
    data.reset_index(inplace=True,drop=True)
    data.drop(['Country Code','Indicator Code'], inplace=True, axis=1)
    data_t = data.transpose()
    return data,data_t

# call the function 
data , data_t = readfile('ads2_data.csv')