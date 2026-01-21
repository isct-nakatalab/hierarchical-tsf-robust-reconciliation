import numpy as np
import pandas as pd


def bottom_up(df, df_base, num_bottom, num_agg, columns_agg):
    print("== START Bottom_Up ==")

    # prepare data
    df_agg = df.iloc[:, :num_agg]
    df_bottom = df.iloc[:, -num_bottom:]

    # calculate summing-matrix
    df_c = pd.DataFrame(
        np.zeros((num_agg, num_bottom)), index=df_agg.columns, columns=df_bottom.columns
    )
    for idx, i in enumerate(df_agg.columns):
        for j in columns_agg[idx]:
            df_c.loc[i, j] = 1
    mtrx_c = df_c.to_numpy()
    mtrx_s = np.concatenate([mtrx_c, np.eye(num_bottom)], axis=0)

    # calculate reconciliation-matrix
    mtrx_p = np.concatenate(
        [np.zeros((num_bottom, num_agg)), np.eye(num_bottom)], axis=1
    )

    # reconciliation
    mtrx_sp = mtrx_s @ mtrx_p
    mtrx_yhat = df_base.to_numpy().T
    mtrx_ytilde = mtrx_sp @ mtrx_yhat

    # convert to DataFrame
    df_bottom_up = pd.DataFrame(mtrx_ytilde.T)
    df_bottom_up.index = df.index
    df_bottom_up.columns = df.columns

    print("== END Bottom_Up ==")

    return df_bottom_up


def top_down(df, df_base, num_bottom, num_agg, columns_agg):
    print("== START Top_Down ==")

    # prepare data
    df_agg = df.iloc[:, :num_agg]
    df_bottom = df.iloc[:, -num_bottom:]

    # calculate summing-matrix
    df_c = pd.DataFrame(
        np.zeros((num_agg, num_bottom)), index=df_agg.columns, columns=df_bottom.columns
    )
    for idx, i in enumerate(df_agg.columns):
        for j in columns_agg[idx]:
            df_c.loc[i, j] = 1
    mtrx_c = df_c.to_numpy()
    mtrx_s = np.concatenate([mtrx_c, np.eye(num_bottom)], axis=0)

    # calculate reconciliation-matrix
    ratio = []
    for c in df_bottom.columns:
        past_ave = df.mean()
        ratio += [past_ave[c] / past_ave.iloc[0]]
    mtrx_p = np.concatenate(
        [
            np.array(ratio).reshape(-1, 1),
            np.zeros((num_bottom, num_agg + num_bottom - 1)),
        ],
        axis=1,
    )

    # reconciliation
    mtrx_sp = mtrx_s @ mtrx_p
    mtrx_yhat = df_base.to_numpy().T
    mtrx_ytilde = mtrx_sp @ mtrx_yhat

    # convert to DataFrame
    df_top_down = pd.DataFrame(mtrx_ytilde.T)
    df_top_down.index = df.index
    df_top_down.columns = df.columns

    print("== END Top_Down ==")

    return df_top_down


def gls(df, df_base, num_bottom, num_agg, columns_agg):
    print("== START GLS_Reconciliation ==")

    # prepare data
    df_agg = df.iloc[:, :num_agg]
    df_bottom = df.iloc[:, -num_bottom:]

    # calculate summing-matrix
    df_c = pd.DataFrame(
        np.zeros((num_agg, num_bottom)), index=df_agg.columns, columns=df_bottom.columns
    )
    for idx, i in enumerate(df_agg.columns):
        for j in columns_agg[idx]:
            df_c.loc[i, j] = 1
    mtrx_c = df_c.to_numpy()
    mtrx_s = np.concatenate([mtrx_c, np.eye(num_bottom)], axis=0)

    # calculate reconciliation-matrix
    mtrx_p = np.linalg.solve(mtrx_s.T @ mtrx_s, mtrx_s.T)

    # reconciliation
    mtrx_sp = mtrx_s @ mtrx_p
    mtrx_yhat = df_base.to_numpy().T
    mtrx_ytilde = mtrx_sp @ mtrx_yhat

    # convert to DataFrame
    df_ols = pd.DataFrame(mtrx_ytilde.T)
    df_ols.index = df.index
    df_ols.columns = df.columns

    print("== END GLS_Reconciliation ==")

    return df_ols


def mint(df, df_base, num_train, num_bottom, num_agg, columns_agg):
    print("== START MinT_Reconciliation ==")

    # prepare data
    df_agg = df.iloc[:, :num_agg]
    df_bottom = df.iloc[:, -num_bottom:]
    df_base_diff = df_base - df
    df_base_diff_train = df_base_diff.iloc[:num_train]

    # estimate variance-covariance matrix (shrinkage)
    mtrx_w = np.cov(df_base_diff_train.to_numpy().T, bias=False)
    mtrx_w_d = np.diag(np.diag(mtrx_w))
    df_e = df_base_diff_train - df_base_diff_train.mean(axis=0)
    mtrx_cov = np.zeros(((num_agg + num_bottom), (num_agg + num_bottom), num_train))
    for i in range(num_agg + num_bottom):
        for j in range(num_agg + num_bottom):
            mtrx_cov[i][j] = (df_e.iloc[:, i] * df_e.iloc[:, j]).to_numpy()
    mtrx_cov_mean = mtrx_cov.mean(axis=-1)
    cov_x = mtrx_cov_mean * num_train / (num_train - 1)
    var_s = np.zeros(cov_x.shape)
    for i in range(num_agg + num_bottom):
        for j in range(num_agg + num_bottom):
            var_s[i][j] = np.sum((mtrx_cov[i][j] - mtrx_cov_mean[i][j]) ** 2)
    var_s = var_s * num_train / ((num_train - 1) ** 3)
    lmd_n = 0
    lmd_d = 0
    for i in range(num_agg + num_bottom):
        for j in range(num_agg + num_bottom):
            if i != j:
                lmd_n += var_s[i][j]
                lmd_d += cov_x[i][j] ** 2
    lmd = lmd_n / lmd_d
    mtrx_w = lmd * mtrx_w_d + (1 - lmd) * mtrx_w

    # calculate summing-matrix
    df_c = pd.DataFrame(
        np.zeros((num_agg, num_bottom)), index=df_agg.columns, columns=df_bottom.columns
    )
    for idx, i in enumerate(df_agg.columns):
        for j in columns_agg[idx]:
            df_c.loc[i, j] = 1
    mtrx_c = df_c.to_numpy()
    mtrx_s = np.concatenate([mtrx_c, np.eye(num_bottom)], axis=0)

    # calculate reconciliation-matrix
    mtrx_j = np.concatenate(
        [np.zeros((num_bottom, num_agg)), np.eye(num_bottom)], axis=1
    )
    mtrx_u = np.concatenate([np.eye(num_agg), -mtrx_c.T], axis=0)
    mtrx_utw = mtrx_u.T @ mtrx_w
    mtrx_p = (
        mtrx_j
        - np.linalg.solve(
            mtrx_utw[:, num_agg:] @ mtrx_u[num_agg:] + mtrx_utw[:, :num_agg],
            mtrx_utw[:, num_agg:] @ mtrx_j.T[num_agg:],
        ).T
        @ mtrx_u.T
    )

    # reconciliation
    mtrx_sp = mtrx_s @ mtrx_p
    mtrx_yhat = df_base.to_numpy().T
    mtrx_ytilde = mtrx_sp @ mtrx_yhat

    # convert to DataFrame
    df_mint = pd.DataFrame(mtrx_ytilde.T)
    df_mint.index = df.index
    df_mint.columns = df.columns

    print("== END MinT_Reconciliation ==")

    return df_mint
