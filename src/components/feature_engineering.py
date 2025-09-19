import os
import sys
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from exception import CustomException
from logger import logging

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,df:pd.dfFrame):
        try:
            logging.info("Starting feature engineering process")

            df['discount_amount'] = df['base_price'] - df['checkout_price']
            
            df['discount_percentage'] = round((df['discount_amount'] / df['base_price']) * 100,2)
            
            df['discount(y/n)'] = [1 if x > 0 else 0 for x in (df['base_price'] - df['checkout_price'])]
            
            df['weekly_base_price_change'] = df.groupby(['meal_id','center_id'])['base_price'].diff().fillna(0)
            
            df['weekly_checkout_price_change'] = df.groupby(['meal_id','center_id'])['checkout_price'].diff().fillna(0)
            
            df['4_week_avg_checkout_price'] = df.groupby(['meal_id', 'center_id'])['checkout_price'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
            
            df['4_week_avg_base_price'] = df.groupby(['meal_id', 'center_id'])['base_price'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
            
            df['week_of_year'] = df['week'].apply(lambda x: x % 52 if x % 52 != 0 else 52)
            
            df['quarter'] = df['week_of_year'].apply(lambda x: (x-1) // 13 + 1)
            
            df['month'] = df['week_of_year'].apply(lambda x: (x-1) // 4 + 1)

            #Creating lag feature
            df = df.sort_values(['meal_id','center_id','week'])
            for lag in [1,2,3,4]:
                df[f'lag_{lag}'] = df.groupby(['meal_id','center_id'])['num_orders'].shift(lag)

            rolling_window = df.groupby(['meal_id','center_id'])['num_orders'].shift(1).rolling(window=4, min_periods=1)

            df['rolling_4_week_mean'] = rolling_window.mean().reset_index(0,drop=True)

            first_week_map = df.groupby(['center_id', 'meal_id'])['week'].min().to_dict()
            df['weeks_on_menu'] = df['week'] - df.set_index(['center_id', 'meal_id']).index.map(first_week_map)

            # Calculate the average checkout price for each category in each week
            avg_price_cat = df.groupby(['week', 'category'])['checkout_price'].transform('mean')
            df['price_vs_category_avg'] = df['checkout_price'] - avg_price_cat

            # Example interaction: promotion AND discount
            df['promo_and_discount'] = df['emailer_for_promotion'] * df['discount_y/n']

            # Example interaction: featured AND discount
            df['featured_and_discount'] = df['homepage_featured'] * df['discount_y/n']

            df['avg_orders_per_meal'] = df.groupby('meal_id')['num_orders'].transform('mean')
            df['avg_orders_per_center'] = df.groupby('center_id')['num_orders'].transform('mean')

        except Exception as e:
            logging.error(f"Error occurred during feature engineering: {e}")
            raise CustomException(e, sys) from e

