#import modules

import numpy as np
import pandas as pd
import scipy
from tabulate import tabulate
from scipy import stats


class ols():

    #init variables

    def __init__(self, dataset, dependent, regressors, cons = True, fixed_effects = [],
    method = 'standard', cluster = []):
        self.dataset = dataset
        self.dependent = dependent
        self.reggressors = regressors
        self.cons = cons
        self.fixed_effects = fixed_effects
        self.method = method
        self.cluster = cluster
        self.cons_array = []
        if self.cons is True:
            # i need it to concatenate variables
            self.cons_array = ['cons']
        else:
            self.cons_array = []
        #get element zero from dependent vector
        self.dep_var = self.dependent[0]


    # prepare de dataset

    def prep_data(self):
        # cerate column for constant if constant is required in the model
        if self.cons is True:
            self.dataset[self.cons_array[0]] = np.ones(len(self.dataset))
        # retrive keys for regressors
        regressors = self.reggressors + self.cons_array
        # create a sub sample of all regressors and dependent
        if len(self.fixed_effects) != 0:
            sub_sample = self.dataset[self.dependent + regressors +
                                              self.fixed_effects].dropna()
            fe_dummies = pd.get_dummies(sub_sample[self.fixed_effects], drop_first=True)
            sub_sample = pd.concat([sub_sample.drop(self.fixed_effects, axis=1),
                                    fe_dummies], axis=1)

        else:
            sub_sample = self.dataset[self.dependent + regressors +
                                              self.fixed_effects].dropna()

        return {'sample': self.dataset, 'sub_sample': sub_sample}

    def coefficients(self):
        X = self.prep_data().get('sub_sample').drop(self.dependent, axis = 1)
        Y = self.prep_data().get('sub_sample')[self.dep_var]
        return np.linalg.inv(X.T@X)@(X.T@Y).to_numpy().reshape((len(X.columns),1))

    def fitted(self):
        return self.prep_data().get('sub_sample').drop(self.dependent, axis = 1)\
               @self.coefficients()

    def residuals(self):
        Y = self.prep_data().get('sub_sample')[self.dep_var].to_numpy()
        Y= Y.reshape((len(self.prep_data().get('sub_sample')),1))
        return Y - self.fitted()

    def AVAR(self):
        X = self.prep_data().get('sub_sample').drop(self.dependent, axis = 1)
        XX_inv = np.linalg.inv(X.to_numpy().T @ X.to_numpy())
        if self.method == 'standard':
            ssr = self.residuals().T@self.residuals()
            dfg = 1/ (len(X) -len(X.columns) )

            sigma_hat = ssr * dfg
            return XX_inv* sigma_hat.values

        elif self.method ==  'heter':
            e_square = (self.residuals()**2).to_numpy().reshape((1, len(self.residuals())))[0]
            e_diag = np.diag(e_square)
            return (XX_inv @ (X.to_numpy().T @ e_diag @ X.to_numpy()) @ XX_inv)

            #reference https://en.wikipedia.org/wiki/Heteroskedasticity-consistent_standard_errors
            # compare with wooldridge examples

        elif self.method == 'cluster':
            #get the sample dataset
            sample = self.prep_data().get('sample')
            # be careful that cluster var is not repeated when it is also usead
            # as fixed effect ???
            keys = self.dependent + self.reggressors + self.cons_array +\
                 self.cluster + [i for i in self.fixed_effects
            if i not in self.dependent + self.reggressors + self.cons_array + self.cluster ]

            #select the subsample from keys and dropna (like stata)
            sample  = sample[keys].dropna()
            #generate dummies from fixed effects
            dummies = pd.get_dummies(sample[self.fixed_effects], drop_first=True)
            #nsert dummies in sample dataset
            sample = pd.concat([sample, dummies], axis = 1)
            #attach residuals column
            sample['residuals'] = self.residuals()
            #take unique clusters
            clusters  = list(set(sample[self.cluster].values.reshape((1, len(sample[self.cluster])))[0]))
            #set self cluster var as index to iterate across rows in the same cluster
            sample = sample.set_index(self.cluster)
            # for loop to iterate across clusters
            shape_sigma = len(self.reggressors) + 1 + len(dummies.columns)
            # initialize a zero sigma matrix. will be the sum of Bs for each cluster
            sigma_matrix = np.zeros((shape_sigma,shape_sigma))
            for c in (clusters):
                # select the subsample for cluster c
                c_sample = sample.loc[c]
                # initialize eps as the vector of residuals of cluster c
                eps_c = c_sample['residuals'].to_numpy().reshape((len(c_sample),1))
                # inirialize the vector of regressors of cluster c
                X_c = c_sample[[i for i in c_sample.columns
                                if i not in self.dependent + self.fixed_effects + self.cluster
                                +['residuals']]].values
                # calculate Bs for each cluster and sum the zero matrix
                sigma_matrix += X_c.T@eps_c@eps_c.T@X_c


            return (XX_inv@sigma_matrix@XX_inv)


    def standard_dev(self):
        return (np.sqrt(np.diagonal(self.AVAR()))).reshape((len(self.coefficients()),1))

    def t_stats(self):
        return (np.round(self.coefficients() / self.standard_dev(),2))

    def p_value(self):
        tvec = self.t_stats()
        dfree = len(self.prep_data().get('sub_sample').drop(self.dependent, axis = 1)) - 1
        return np.round(scipy.stats.norm.sf(abs(tvec)) * 2,2)

    def confidence(self):
        betas = self.coefficients()
        std = self.standard_dev()
        low = betas - std*1.96
        high = betas + std*1.96

        return {'low': low, 'high': high}

    def summary(self):
        header = [self.dependent[0], 'coefficient', 'se', 't', 'p_value', 'low 95', 'high 95']
        table = []
        vars = self.reggressors + self.cons_array
        def reshaping(array):
            return array[0:len(vars)].reshape((1,len(vars)))[0]

        vec = [vars, reshaping(self.coefficients()), reshaping(self.standard_dev()),
        reshaping(self.t_stats()),reshaping(self.p_value()), reshaping(self.confidence().get('low')),
        reshaping(self.confidence().get('high'))]
        vec = list(map(list, zip(*vec)))
        print('OLS Regression')

        print('------------------------------------------------------------------------------------')
        print(tabulate(vec, headers=header))
        print('------------------------------------------------------------------------------------')
        return ''


class two_sls():

    def __init__(self, dataset, dependent, regressors, endogenous, instruments,
                 cons = True, fixed_effects = []):
        self.dataset = dataset
        self.dependent = dependent
        self.regressors = regressors
        self.endogenous = endogenous
        self.instruments = instruments
        self. cons = cons
        if self.cons is True:
            self.cons_arr = ['cons']
        else:
            self.cons_arr = []
        self.fixed_effects = fixed_effects

    def retrive_data(self):
        data = self.dataset[self.dependent + self.regressors +
                            self.endogenous + self.instruments].dropna()
        if len(self.fixed_effects) == 0:
            return data
        else:
            dummies = pd.get_dummies(self.fixed_effects, drop_first= True)
            return pd.concat([data, dummies], axis = 1)


    def first_stage(self):
        # define list of regressors for first stage

        fs_regressors = [i for i in self.retrive_data().columns.tolist() if
                         i not in self.dependent + self.endogenous]

        #initializing the linear regression for the first stage
        betas_matrix = np.zeros((len(fs_regressors)+1
                                 , 1))
        fitted_matrix = np.zeros((len(self.retrive_data()),1))

        for end in range(len(self.endogenous)):
            fs_obj = ols(dataset=self.retrive_data(), regressors=fs_regressors, dependent=[self.endogenous[end]]
                         , fixed_effects=self.fixed_effects)

            betas_matrix= np.concatenate([betas_matrix, fs_obj.coefficients()], axis=1)
            fitted_matrix =  np.concatenate([fitted_matrix, fs_obj.fitted()], axis=1)


        betas_matrix = betas_matrix[:,1:]
        fitted_matrix = fitted_matrix[:, 1:]
        return {'betas' : betas_matrix, 'fitted' : fitted_matrix}


    def second_stage(self):
        # attach first stage fitted values in original datsaet
        first_dataset = self.retrive_data()
        for end in range(len(self.endogenous)):
            first_dataset[f'1stage_{self.endogenous[end]}'] = self.first_stage().get('fitted')[:,end]


        ss_xs = self.regressors + [i for i in first_dataset.columns.tolist() if
                                       i not in self.dependent + self.endogenous
                                       + self.instruments + self.regressors]


        # initialize model for 2nd stage regression
        model_ss = ols(first_dataset, dependent=self.dependent, regressors=ss_xs, fixed_effects=self.fixed_effects)
        # computing beta coefficients
        b_hat = model_ss.coefficients()
        Z = first_dataset[self.regressors + self.instruments + self.cons_arr]
        # get matrix with x variables
        sub_sample_2nd = ols(first_dataset, dependent=self.dependent, regressors=self.regressors + self.endogenous,
                             fixed_effects=self.fixed_effects).prep_data() \
            .get('sub_sample')
        X_2nd = sub_sample_2nd.drop(self.dependent, axis=1)
        Y = (sub_sample_2nd[self.dependent[0]].values)
        Y = Y.reshape((len(Y), 1))

        # computing fitted values for 2nd stage residuals
        XB_hat = (X_2nd @ b_hat).values
        XB_hat = XB_hat.reshape((len(XB_hat), 1))
        # computing 2nd stage residuals
        residual_2nd = Y - XB_hat
        # computing sigma 2nd stage
        sigma_hat = (residual_2nd.T @ residual_2nd) / (
                len(residual_2nd) - len(self.regressors + self.endogenous))
        # computing 2nd stage variance covariance matrix
        X_hat_second = (Z.T @ X_2nd).values
        X_hat_first = (np.linalg.inv(Z.T @ Z))
        X_hat = Z @ (X_hat_first @ X_hat_second)
        vars_matrix = sigma_hat * np.linalg.inv(X_hat.T @ X_hat)
        std_var = np.sqrt(np.diagonal(vars_matrix)).reshape((len(b_hat), 1))
        # computing t statistic
        t = np.round(model_ss.coefficients() / std_var, 2)
        # computing p value
        dfree = len(model_ss.prep_data().get('sub_sample')) - 1
        p_val = np.round(scipy.stats.t.sf(abs(t), df=dfree) * 2, 3)

        # computing confidence bands
        low = model_ss.coefficients() - (std_var * 1.96)
        high = model_ss.coefficients() + (std_var * 1.96)

        return {'beta': model_ss.coefficients(), 'std': std_var, 't': t, 'p': p_val, 'low': low, 'high': high,
                'var_matrix': vars_matrix}

    def summary(self):
        print('2SLS Regression')
        header = [self.dependent, 'coefficient', 'se', 't', 'p_value', 'low 95', 'high 95']
        table = []
        vars = self.regressors + self.endogenous + self.cons_arr

        vec = [vars, self.second_stage().get('beta'), self.second_stage().get('std'), self.second_stage().get('t'),
        self.second_stage().get('p'), self.second_stage().get('low'),  self.second_stage().get('high')]
        vec = list(map(list, zip(*vec)))

        print('-------------------------------------------------------------------------------')
        print(tabulate(vec, headers=header))
        print('-------------------------------------------------------------------------------')
        print(f'Endogenous variable: {self.endogenous}')
        print(f'Instruments: {self.regressors + self.instruments} ')
        print('-------------------------------------------------------------------------------')


        return ''

