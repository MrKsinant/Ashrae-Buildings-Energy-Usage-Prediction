#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#######################################################
# PROGRAMMER: Pierre-Antoine Ksinant                  #
# DATE CREATED: 27/03/2020                            #
# REVISED DATE: -                                     #
# PURPOSE: General framework for modeling the problem #
#######################################################


##################
# Needed imports #
##################

import pandas as pd
from numpy import sqrt, absolute
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split


#################################
# Create an illustration sample #
#################################

def create_illustration_sample(dataset, building_id, week=42):
    """ Create an illustration sample """
    
    # Isolate sample data:
    sample = dataset.loc[(dataset['building_id']==building_id) & (dataset['week']==week)]
    
    # Create various sample aspects:
    timestamp_sample = pd.to_datetime(sample['timestamp'])
    target_sample = sample['meter_reading']
    extracted_target_sample = target_sample.values
    features_to_drop = ['building_id',
                        'meter',
                        'timestamp',
                        'site_id',
                        'hour',
                        'weekday',
                        'week',
                        'month',
                        'meter_reading']
    features_sample = sample.drop(features_to_drop, axis=1)
    extracted_features_sample = features_sample.values
    
    # Remove sample from dataset:
    dataset_2 = dataset.drop(dataset[(dataset['building_id']==building_id) & (dataset['week']==week)].index)
    
    # Return results:
    return dataset_2, timestamp_sample, extracted_target_sample, extracted_features_sample


############################################
# Create datasets for training and testing #
############################################

def create_training_testing_datasets(dataset):
    """ Create shuffled datasets """
    
    # Clean up the dataset:
    features_to_drop = ['building_id',
                        'meter',
                        'timestamp',
                        'site_id',
                        'hour',
                        'weekday',
                        'week',
                        'month']
    dataset_2 = dataset.drop(features_to_drop, axis=1)
    
    # Separate target from features in the dataset:
    target = dataset_2['meter_reading']
    features = dataset_2.drop('meter_reading', axis=1)
    
    # Extract target and features:
    extracted_target = target.values
    extracted_features = features.values
    
    # Shuffle and split the data into training and testing datasets:
    X_train, X_test, y_train, y_test = train_test_split(extracted_features,
                                                        extracted_target,
                                                        test_size=0.2,
                                                        random_state=42)
    
    # Return results:
    return X_train, X_test, y_train, y_test


##########################
# Calculate RMSLE metric #
##########################

def calculate_rmsle(y_true, y_pred):
    """ Calculate RMSLE metric """
    
    # Calculate metric:
    rmsle = sqrt(mean_squared_log_error(y_true, absolute(y_pred)))
    
    # Return result:
    return rmsle