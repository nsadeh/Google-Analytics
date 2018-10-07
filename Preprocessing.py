import pandas
import json
import os
import numpy as np
from functools import reduce
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class Preprocessing(object):

    TRAIN_PATH = 'C:\\Users\\nsadeh\\Documents\\Kaggle Projects\\Data Sets\\Google Analytics\\train.csv'
    TEST_PATH = 'DUMMY'
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    DEVICE_OS = ['Windows', 'Macintosh', 'Android', 'iOS', 'Linux', 'Chrome OS']
    DEVICE_BROWSER = ['Chrome', 'Safari', 'Firefox', 'Internet Explorer', 'Edge', 'Opera Mini', 'Opera']

    def __init__(self,
                 train_path=TRAIN_PATH,
                 test_path=TEST_PATH):
        self.train_path = train_path
        self.test_path = test_path
        self.pageview_scaler = None
        self.device_encoder = None

    def load_data(self,
                  training=True):
        """
        loads training/testing data
        :param training: boolean: whether we be loading train/test data
        :return: 6 pandas.DataFrame objects of the 5 tables in this set
        """
        path = self.train_path if training else self.test_path
        try:
            assert os.path.exists(path)
        except AssertionError:
            raise Exception('Path to {} file does not exist'.format('training'
                                                                    if training
                                                                    else 'testing'))
        print('Loading data')
        train_raw = pandas.read_csv(path,
                                    dtype={'fullVisitorId': 'str'})
        device = self.read_json(train_raw, 'device')
        geo_network = self.read_json(train_raw, 'geoNetwork')
        totals = self.read_json(train_raw, 'totals')
        traffic_source = self.read_json(train_raw, 'trafficSource')
        train_raw.drop(self.JSON_COLUMNS,
                       axis=1,
                       inplace=True)
        print('Data loaded')
        return train_raw, device, geo_network, totals, traffic_source

    def read_json(self, data, column):
        """
        Parses the json columns
        :param data: pandas.DataFrame with the column
        :param column: column of JSON
        :return: pandas.DataFrame with what's in the JSON column
        """
        try:
            assert isinstance(data, pandas.DataFrame)
            assert isinstance(column, str)
            assert column in data.columns
        except AssertionError:
            raise Exception('Wrong inputs in read_json')
        json_var = data[column]
        record = json_var.map(json.loads).values.tolist()
        json_df = pandas.DataFrame.from_records(record)
        return json_df

    @staticmethod
    def remove_consts(data):
        assert isinstance(data, pandas.DataFrame)
        constants = [col
                     for col in data.columns
                     if data[col].nunique(dropna=False) == 1]
        return data.drop(constants, axis=1)

    @staticmethod
    def month(date):
        """
        Transforms a string of the format yyyymmdd to one of the calendar months
        :param date: string of form yyyymmdd
        :return: month as a name
        """
        month = int(str(date)[4:6]) - 1
        months = ['January',
                  'February',
                  'March',
                  'April',
                  'May',
                  'June',
                  'July',
                  'August',
                  'September',
                  'October',
                  'November',
                  'December']
        return months[month]

    def process_totals(self, totals, training=True):
        """
        Shrink number range into something more manageable
        :param totals: dataFrame with columns
        :param training: whether to fit a transformation or use an existing one
        :return: pageviews, transaction_value (target)
        TODO: still have NaN issue
        """
        totals = self.remove_consts(totals)
        pagviews = np.log(totals.pageviews.values)
        if training:
            scaler = MinMaxScaler()
            normed_pageviews = scaler.fit(pagviews)
            self.pageview_scaler = scaler
        else:
            assert self.pageview_scaler, 'pageview scaler not initialized'
            normed_pageviews = self.pageview_scaler.transform(pagviews)
        return np.log(totals.transactionRevenue.values), normed_pageviews

    def process_device(self, device, training=True):
        device = device[['browser', 'isMobile', 'operatingSystem']]
        device = reduce(lambda var, val: self.to_categorical(device,
                                                             ['browser', 'isMobile', 'operatingSystem'],
                                                             [self.DEVICE_BROWSER, ['True', 'False'], self.DEVICE_OS]))
        if training:
            encoder = OneHotEncoder()
            encoded_device = encoder.fit_transform(device.as_matrix())
            self.device_encoder = encoder
        else:
            assert self.device_encoder, 'pageview scaler not initialized'
            encoded_device = self.pageview_scaler.transform(device)
        return pandas.DataFrame(encoded_device

    @staticmethod
    def to_categorical(df,
                       var,
                       allowed_vals,
                       replace='Other'):
        """
        Converts DataFrame column to categorical with allowed values
        :param df: DataFrame
        :param var: column in DataFrame
        :param allowed_vals: values allowed
        :return: DataFrame where column var is categorical with len(allowed_vals) + 1 cats
        """
        vals = df[var].values
        updated_vals = [val
                        if val in allowed_vals
                        else replace
                        for val in vals]
        df[var] = pandas.Series(updated_vals).astype('category')
        return df




if __name__ == '__main__':
    pre = Preprocessing()
    train_raw, device, geo_network, totals, traffic_source = pre.load_data()
    print(traffic_source)
