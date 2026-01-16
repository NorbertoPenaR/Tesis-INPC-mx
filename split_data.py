import pandas as pd


class Split:

    @staticmethod
    def split_data(data=None, train_years=0, months_val_test=1, date=None, min_rows=50):

        # select minimum quantity of rows
        if data.shape[0] < min_rows:
            months_val_test = 1

        # define cut-offs
        months = 2 * months_val_test

        cutoff_train = data[date].max() - pd.DateOffset(months=months)
        cutoff_val_test = data[date].max() - pd.DateOffset(months=months / 2)
        fil_val = (data[date] > cutoff_train) & (data[date] <= cutoff_val_test)

        # split
        x_val = data[fil_val]
        x_test = data[data[date] > cutoff_val_test]

        if train_years == 0:
            x_train = data[data[date] <= cutoff_train]

        elif train_years != 0:
            x_train = data[
                (data[date] >= data[date].max() - pd.DateOffset(years=train_years)) & (data[date] <= cutoff_train)
            ]
        else:
            x_train = 0

        return x_train, x_val, x_test






