import pandas
import numpy as np


class ModelEvaluation(object):

    def __init__(self,
                 visitor_id,
                 data,
                 predicted_revenue,
                 labeled_revenue):
        """
        Class to evaluate the model success
        :param visitor_id: pandas.Series with visitor ids
        :param data: pandas.DataFrame with corresponding mapped variables
        :param predicted_revenue: pandas.Series with corresponding prediction by model
        :param labeled_revenue: pandas.Series with corresponding labeled revenue
        """
        try:
            assert isinstance(data, pandas.DataFrame)
            assert all(isinstance(var, pandas.Series) for var in [visitor_id,
                                                                  predicted_revenue,
                                                                  labeled_revenue])
        except AssertionError:
            raise TypeError('Incorrect parameters passed to ModelEvaluation constructor')
        self.visitor_id = visitor_id
        self.data = data
        self.predicted_revenue = predicted_revenue
        self.labeled_revenue = labeled_revenue

    def agg_transactions(self):
        """
        test this idea: (pseudo-Python)
        revenue = pandas.concat([visitor_id, predicted_revenue, labeled_revenue])
        aggregated = revenue.groupby(['fullVisitorId']).sum()
        :return: pandas.DataFrame with aggregated transaction values per fullVisitorId
        """
        unique_ids = set(self.visitor_id.values)
        pred = [sum(self.predicted_revenue[self.visitor_id == idx].values) for idx in unique_ids]
        labeled = [sum(self.labeled_revenue[self.visitor_id == idx].values) for idx in unique_ids]
        agg_transactions = pandas.DataFrame({'fullVisitorId': list(unique_ids),
                                             'predictedRevenue': pred,
                                             'labeledRevenue': labeled})
        return agg_transactions
