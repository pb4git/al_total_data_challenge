from package.utilities import any_diff_weights, any_diff_weights_no_zero
import numpy as np
import pandas as pd

# def shift_fill(df, shifts):
#     new_df = df
#     if shifts > 0:
#         for i in range(shifts):
#             new_df = new_df.shift(1)
#             new_df.iloc[0,:] = df.iloc[0,:]
#     if shifts < 0:
#         for i in range(-shifts):
#             new_df = new_df.shift(-1)
#             new_df.iloc[-1,:] = df.iloc[-1,:]
#     return new_df


def make_derivative(df, start, end, ndiv):
    coeffs = any_diff_weights(start, end, ndiv)
    df_diff = df.copy()
    df_diff[:] = 0
    for i, c in zip(range(start, 1+end), coeffs):
        df_diff += df.shift(-i) * c
    return df_diff

def make_derivative_no_zero(df, start, end, ndiv):
    # used to interpolate, function name could be improved
    coeffs = any_diff_weights_no_zero(start, end, ndiv)
    df_diff = df.copy()
    df_diff[:] = 0
    for i, c in zip([x for x in range(start, 1+end) if x != 0], coeffs):
        df_diff += df.shift(-i) * c
    return df_diff

def make_OLS_derivative(df, start, end):
    n = 1 + end - start
    sum_xi = df.copy()
    sum_xi[:] = 0
    sum_yi =    sum_xi.copy()
    sum_xiyi =  sum_xi.copy()
    sum_xixi =  sum_xi.copy()
    r2 =        sum_xi.copy()
    for i in range(start, 1+end):
        sum_xi      += i
        sum_yi      += df.shift(-i)
        sum_xiyi    += i*df.shift(-i)
        sum_xixi    += i**2
    beta_hat = (n*sum_xiyi - sum_xi*sum_yi) / (n*sum_xixi - sum_xi**2)
    alpha_hat = (sum_yi - beta_hat*sum_xi) / n
    for i in range(start, 1+end):
        r2 += (df.shift(-i) - (alpha_hat + beta_hat*i))**2
    r2 = np.sqrt(r2 / n) #not actually r2, we take the square root
    return alpha_hat, beta_hat, r2 


def make_sliding(pivoted_df, base_name, window_size):
    
    mean = pivoted_df.copy()
    shifted_pivoted_df = pivoted_df.copy()
    for j in range(0, window_size//2):
        shifted_pivoted_df = shifted_pivoted_df.shift(1)
        shifted_pivoted_df.iloc[0,:] = pivoted_df.iloc[0,:]
        mean += shifted_pivoted_df
    shifted_pivoted_df = pivoted_df.copy()
    for j in range(0, window_size//2):
        shifted_pivoted_df = shifted_pivoted_df.shift(-1)
        shifted_pivoted_df.iloc[-1,:] = pivoted_df.iloc[-1,:]
        mean += shifted_pivoted_df
    mean /= window_size
    
    std = pivoted_df.rolling(window_size).std().shift(-(window_size//2))
    for j in range(0, window_size//2):
        std.iloc[j,:] = pivoted_df.rolling(window_size).std().shift(-(window_size-1)+j).iloc[j,:]
    for j in range(0, window_size//2):
        std.iloc[-1-j,:] = pivoted_df.rolling(window_size).std().shift(-j).iloc[-1-j,:]
    
    min = pivoted_df.copy()
    max = pivoted_df.copy()
    shifted_pivoted_df = pivoted_df.copy()
    for j in range(0, window_size//2):
        shifted_pivoted_df = shifted_pivoted_df.shift(1)
        shifted_pivoted_df.iloc[0,:] = pivoted_df.iloc[0,:]
        min = np.minimum(min, shifted_pivoted_df)
        max = np.maximum(max, shifted_pivoted_df)
    shifted_pivoted_df = pivoted_df.copy()
    for j in range(0, window_size//2):
        shifted_pivoted_df = shifted_pivoted_df.shift(-1)
        shifted_pivoted_df.iloc[-1,:] = pivoted_df.iloc[-1,:]
        min = np.minimum(min, shifted_pivoted_df)
        max = np.maximum(max, shifted_pivoted_df)

    std = pd.concat([std], axis=1, keys=[base_name+'_std_w'+str(window_size)])
    mean = pd.concat([mean], axis=1, keys=[base_name+'_mean_w'+str(window_size)])
    min = pd.concat([min], axis=1, keys=[base_name+'_min_w'+str(window_size)])
    max = pd.concat([max], axis=1, keys=[base_name+'_max_w'+str(window_size)])

    return [std, mean, min, max]


def make_derivatives(pivoted_df, base_name):
    derivatives_df = []
    
    #center first derivative
    temp = make_derivative(pivoted_df, -3, 3, 1)
    temp.iloc[0,:] = make_derivative(pivoted_df,  0, 3, 1).iloc[0,:]
    temp.iloc[1,:] = make_derivative(pivoted_df, -1, 3, 1).iloc[1,:]
    temp.iloc[2,:] = make_derivative(pivoted_df, -2, 3, 1).iloc[2,:]
    temp.iloc[-3,:] = make_derivative(pivoted_df, -3, 2, 1).iloc[-3,:]
    temp.iloc[-2,:] = make_derivative(pivoted_df, -3, 1, 1).iloc[-2,:]
    temp.iloc[-1,:] = make_derivative(pivoted_df, -3, 0, 1).iloc[-1,:]
    derivatives_df.append(pd.concat([temp], axis=1, keys=[base_name+'_d1_cl']))

    #tight center first derivative
    temp = make_derivative(pivoted_df, -1, 1, 1)
    temp.iloc[0,:] = make_derivative(pivoted_df,  0, 1, 1).iloc[0,:]
    temp.iloc[-1,:] = make_derivative(pivoted_df, -1, 0, 1).iloc[-1,:]
    derivatives_df.append(pd.concat([temp], axis=1, keys=[base_name+'_d1_ct']))

    #center second derivative
    temp = make_derivative(pivoted_df, -3, 3, 2)
    temp.iloc[0,:] = make_derivative(pivoted_df,  0, 3, 2).iloc[0,:]
    temp.iloc[1,:] = make_derivative(pivoted_df, -1, 3, 2).iloc[1,:]
    temp.iloc[2,:] = make_derivative(pivoted_df, -2, 3, 2).iloc[2,:]
    temp.iloc[-3,:] = make_derivative(pivoted_df, -3, 2, 2).iloc[-3,:]
    temp.iloc[-2,:] = make_derivative(pivoted_df, -3, 1, 2).iloc[-2,:]
    temp.iloc[-1,:] = make_derivative(pivoted_df, -3, 0, 2).iloc[-1,:]
    derivatives_df.append(pd.concat([temp], axis=1, keys=[base_name+'_d2_cl']))

    #tight center second derivative
    temp = make_derivative(pivoted_df, -1, 1, 2)
    temp.iloc[0,:] = make_derivative(pivoted_df,  0, 2, 2).iloc[0,:]
    temp.iloc[-1,:] = make_derivative(pivoted_df, -2, 0, 2).iloc[-1,:]
    derivatives_df.append(pd.concat([temp], axis=1, keys=[base_name+'_d2_ct']))

    #center third derivative
    temp = make_derivative(pivoted_df, -3, 3, 3)
    temp.iloc[0,:] = make_derivative(pivoted_df,  0, 4, 3).iloc[0,:]
    temp.iloc[1,:] = make_derivative(pivoted_df, -1, 3, 3).iloc[1,:]
    temp.iloc[2,:] = make_derivative(pivoted_df, -2, 3, 3).iloc[2,:]
    temp.iloc[-3,:] = make_derivative(pivoted_df, -3, 2, 3).iloc[-3,:]
    temp.iloc[-2,:] = make_derivative(pivoted_df, -3, 1, 3).iloc[-2,:]
    temp.iloc[-1,:] = make_derivative(pivoted_df, -4, 0, 3).iloc[-1,:]
    derivatives_df.append(pd.concat([temp], axis=1, keys=[base_name+'_d3_cl']))

    #left first derivative
    temp = make_derivative(pivoted_df, -3, 1, 1)
    temp.iloc[0,:] = make_derivative(pivoted_df,  0, 1, 1).iloc[0,:]
    temp.iloc[1,:] = make_derivative(pivoted_df, -1, 1, 1).iloc[1,:]
    temp.iloc[2,:] = make_derivative(pivoted_df, -2, 1, 1).iloc[2,:]
    temp.iloc[-1,:] = make_derivative(pivoted_df, -1, 0, 1).iloc[-1,:]
    derivatives_df.append(pd.concat([temp], axis=1, keys=[base_name+'_d1_ll']))

    #right first derivative
    temp = make_derivative(pivoted_df, -1, 3, 1)
    temp.iloc[0,:] = make_derivative(pivoted_df,  0, 1, 1).iloc[0,:]
    temp.iloc[-3,:] = make_derivative(pivoted_df, -1, 2, 1).iloc[-3,:]
    temp.iloc[-2,:] = make_derivative(pivoted_df, -1, 1, 1).iloc[-2,:]
    temp.iloc[-1,:] = make_derivative(pivoted_df, -1, 0, 1).iloc[-1,:]
    derivatives_df.append(pd.concat([temp], axis=1, keys=[base_name+'_d1_rl']))

    # interpolate
    # not an actual derivative, but since the method is nearly the same I leave it here
    temp  = make_derivative_no_zero(pivoted_df, -3, 3, 0)
    temp.iloc[-3,:] = (0.1*pivoted_df + 0.9*make_derivative_no_zero(pivoted_df, -3, 2, 0)).iloc[-3,:]
    temp.iloc[-2,:] = (0.3*pivoted_df + 0.7*make_derivative_no_zero(pivoted_df, -3, 1, 0)).iloc[-2,:]
    temp.iloc[-1,:] = (0.8*pivoted_df + 0.2*make_derivative_no_zero(pivoted_df, -3, 0, 0)).iloc[-1,:]
    temp.iloc[2,:] = (0.1*pivoted_df + 0.9*make_derivative_no_zero(pivoted_df, -2, 3, 0)).iloc[2,:]
    temp.iloc[1,:] = (0.3*pivoted_df + 0.7*make_derivative_no_zero(pivoted_df, -1, 3, 0)).iloc[1,:]
    temp.iloc[0,:] = (0.8*pivoted_df + 0.2*make_derivative_no_zero(pivoted_df,  0, 3, 0)).iloc[0,:]
    derivatives_df.append(pd.concat([temp], axis=1, keys=[base_name+'_interp']))

    return derivatives_df


def make_OLS_derivatives(pivoted_df, base_name):
    derivatives_df = []
    
    alpha_hat, beta_hat, r2  = make_OLS_derivative(pivoted_df, -2, 2)

    temp_alpha_hat, temp_beta_hat, temp_r2  = make_OLS_derivative(pivoted_df, 0, 3)
    alpha_hat.iloc[0,:] =  temp_alpha_hat.iloc[0, :]
    beta_hat.iloc[0,:] =   temp_beta_hat.iloc[0, :]
    r2.iloc[0,:] =         temp_r2.iloc[0, :]

    temp_alpha_hat, temp_beta_hat, temp_r2  = make_OLS_derivative(pivoted_df, -1, 3)
    alpha_hat.iloc[1,:] =  temp_alpha_hat.iloc[1, :]
    beta_hat.iloc[1,:] =   temp_beta_hat.iloc[1, :]
    r2.iloc[1,:] =         temp_r2.iloc[1, :]

    temp_alpha_hat, temp_beta_hat, temp_r2  = make_OLS_derivative(pivoted_df, -3, 1)
    alpha_hat.iloc[-2,:] =  temp_alpha_hat.iloc[-2, :]
    beta_hat.iloc[-2,:] =   temp_beta_hat.iloc[-2, :]
    r2.iloc[-2,:] =         temp_r2.iloc[-2, :]

    temp_alpha_hat, temp_beta_hat, temp_r2  = make_OLS_derivative(pivoted_df, -3, 0)
    alpha_hat.iloc[-1,:] =  temp_alpha_hat.iloc[-1, :]
    beta_hat.iloc[-1,:] =   temp_beta_hat.iloc[-1, :]
    r2.iloc[-1,:] =         temp_r2.iloc[-1, :]

    derivatives_df.append(pd.concat([alpha_hat], axis=1, keys=[base_name+'_alphaOLS']))
    derivatives_df.append(pd.concat([beta_hat], axis=1, keys=[base_name+'_betaOLS']))
    derivatives_df.append(pd.concat([r2], axis=1, keys=[base_name+'_r2OLS']))

    return derivatives_df


    
# def make_derivative_fill(df, start, end, ndiv):
#     coeffs = any_diff_weights(start, end, ndiv)
#     df_diff = df.copy()
#     df_diff[:] = 0
#     for i, c in zip(range(start, 1+end), coeffs):
#         df_diff += shift_fill(df, -i) * c
#     return df_diff

# def make_derivative_fill_no_zero(df, start, end, ndiv):
#     coeffs = any_diff_weights_no_zero(start, end, ndiv)
#     df_diff = df.copy()
#     df_diff[:] = 0
#     for i, c in zip([x for x in range(start, 1+end) if x != 0], coeffs):
#         df_diff += shift_fill(df, -i) * c
#     return df_diff