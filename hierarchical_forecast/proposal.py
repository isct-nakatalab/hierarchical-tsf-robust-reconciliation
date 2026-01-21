import cvxpy as cp
import numpy as np
import pandas as pd


def var_matrix(
    df_base_diff_train, num_train, num_bottom, num_agg, num_bootstrap, alpha
):
    print("> estimate variance-covariance matrix...")

    # calculate shrinkage intensity parameter
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

    # sampling
    mtrx_w_sample = []
    for i in range(num_bootstrap):
        rng = np.random.default_rng(i)
        idx = rng.choice(num_train, size=num_train, replace=True)
        df_base_diff_train_sample = df_base_diff_train.iloc[idx]
        mtrx_w = np.cov(df_base_diff_train_sample.to_numpy().T, bias=False)
        mtrx_w_d = np.diag(np.diag(mtrx_w))
        mtrx_w = lmd * mtrx_w_d + (1 - lmd) * mtrx_w
        mtrx_w = np.linalg.pinv(mtrx_w)
        mtrx_w_sample += [mtrx_w]

    # calculate upper and lower bounds
    mtrx_wl = np.percentile(mtrx_w_sample, 50 * (1 - alpha), axis=0)
    mtrx_wu = np.percentile(mtrx_w_sample, 50 * (1 + alpha), axis=0)

    return mtrx_wl, mtrx_wu


def robust_opt(
    df,
    df_base,
    num_train,
    num_bottom,
    num_agg,
    columns_agg,
    num_bootstrap,
    alpha,
):
    # prepare data
    df_agg = df.iloc[:, :num_agg]
    df_bottom = df.iloc[:, -num_bottom:]
    df_base_diff = df_base - df
    df_base_diff_train = df_base_diff.iloc[:num_train]

    # set upper and lower bounds
    mtrx_wl, mtrx_wu = var_matrix(
        df_base_diff_train, num_train, num_bottom, num_agg, num_bootstrap, alpha
    )

    # calculate summing-matrix
    df_c = pd.DataFrame(
        np.zeros((num_agg, num_bottom)), index=df_agg.columns, columns=df_bottom.columns
    )
    for idx, i in enumerate(df_agg.columns):
        for j in columns_agg[idx]:
            df_c.loc[i, j] = 1
    mtrx_c = df_c.to_numpy()
    mtrx_s = np.concatenate([mtrx_c, np.eye(num_bottom)], axis=0)

    # convert to Numpy array
    mtrx_y = df.to_numpy()
    mtrx_yhat = df_base.to_numpy()

    # scale coefficients
    scale_weight_w = (np.mean(np.abs(mtrx_wl)) + np.mean(np.abs(mtrx_wu))) / 2
    scale_weight_y = (np.mean(np.abs(mtrx_y)) + np.mean(np.abs(mtrx_yhat))) / 2
    mtrx_wl = mtrx_wl / scale_weight_w
    mtrx_wu = mtrx_wu / scale_weight_w
    mtrx_y = mtrx_y / scale_weight_y
    mtrx_yhat = mtrx_yhat / scale_weight_y
    print(f"ScaleWeight_W: {scale_weight_w:.2f}")
    print(f"ScaleWeight_y: {scale_weight_y:.2f}")

    # calculate reconciliation-matrix
    print("-- start solving_optimization_problem --")
    # variables
    print("> generate variables...")
    p = cp.Variable((num_bottom, (num_agg + num_bottom)))
    xu = cp.Variable(((num_agg + num_bottom), (num_agg + num_bottom)), PSD=True)
    xl = cp.Variable(((num_agg + num_bottom), (num_agg + num_bottom)), PSD=True)
    # objective
    print("> generate objective...")
    objective = cp.Minimize(cp.trace(mtrx_wu.T @ xu) - cp.trace(mtrx_wl.T @ xl))
    # constraints
    print("> generate constraints...")
    residual = []
    for t in range(num_train):
        residual += [mtrx_y[t] - mtrx_s @ p @ mtrx_yhat[t]]
    residual = cp.vstack(residual).T
    top = cp.hstack([xu - xl, residual])
    bottom = cp.hstack([residual.T, cp.Constant(np.eye(num_train))])
    sdp_matrix = cp.vstack([top, bottom])
    constraints = [
        sdp_matrix >> 0,
        xu >= 0,
        xl >= 0,
    ]
    # opimal solution
    print("> solve problem...")
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK, verbose=True)
    mtrx_p = np.array(p.value)
    print("-- end solving_optimization_problem --")

    # reconciliation
    mtrx_sp = mtrx_s @ mtrx_p
    mtrx_yhat = df_base.to_numpy().T
    mtrx_ytilde = mtrx_sp @ mtrx_yhat

    # convert to DataFrame
    df_robust = pd.DataFrame(mtrx_ytilde.T)
    df_robust.index = df.index
    df_robust.columns = df.columns

    return df_robust


def validation_alpha(
    df_train,
    df_base_train,
    num_train,
    num_bottom,
    num_agg,
    columns_agg,
    num_bootstrap,
    alpha,
):
    # prepare data
    df_agg = df_train.iloc[:, :num_agg]
    df_bottom = df_train.iloc[:, -num_bottom:]
    df_base_diff_train = df_base_train - df_train

    # set upper and lower bounds
    mtrx_wl, mtrx_wu = var_matrix(
        df_base_diff_train,
        num_train,
        num_bottom,
        num_agg,
        num_bootstrap,
        alpha,
    )

    # calculate summing-matrix
    df_c = pd.DataFrame(
        np.zeros((num_agg, num_bottom)), index=df_agg.columns, columns=df_bottom.columns
    )
    for idx, i in enumerate(df_agg.columns):
        for j in columns_agg[idx]:
            df_c.loc[i, j] = 1
    mtrx_c = df_c.to_numpy()
    mtrx_s = np.concatenate([mtrx_c, np.eye(num_bottom)], axis=0)

    # convert to Numpy array
    mtrx_y = df_train.to_numpy()
    mtrx_yhat = df_base_train.to_numpy()

    # scale coefficients
    scale_weight_w = (np.mean(np.abs(mtrx_wl)) + np.mean(np.abs(mtrx_wu))) / 2
    scale_weight_y = (np.mean(np.abs(mtrx_y)) + np.mean(np.abs(mtrx_yhat))) / 2
    mtrx_wl = mtrx_wl / scale_weight_w
    mtrx_wu = mtrx_wu / scale_weight_w
    mtrx_y = mtrx_y / scale_weight_y
    mtrx_yhat = mtrx_yhat / scale_weight_y
    print(f"ScaleWeight_W: {scale_weight_w:.2f}")
    print(f"ScaleWeight_y: {scale_weight_y:.2f}")

    # calculate reconciliation-matrix
    print("-- start solving_optimization_problem --")
    # variables
    print("> generate variables...")
    p = cp.Variable((num_bottom, num_agg + num_bottom))
    xu = cp.Variable((num_agg + num_bottom, num_agg + num_bottom), PSD=True)
    xl = cp.Variable((num_agg + num_bottom, num_agg + num_bottom), PSD=True)
    # objective
    print("> generate objective...")
    objective = cp.Minimize(cp.trace(mtrx_wu.T @ xu) - cp.trace(mtrx_wl.T @ xl))
    # constraints
    print("> generate constraints...")
    residual = []
    for t in range(num_train):
        residual += [mtrx_y[t] - mtrx_s @ p @ mtrx_yhat[t]]
    residual = cp.vstack(residual).T
    top = cp.hstack([xu - xl, residual])
    bottom = cp.hstack([residual.T, cp.Constant(np.eye(num_train))])
    sdp_matrix = cp.vstack([top, bottom])
    constraints = [
        sdp_matrix >> 0,
        xu >= 0,
        xl >= 0,
    ]
    # optimal solution
    print("> solve problem...")
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cp.MOSEK, verbose=True)
        mtrx_p = np.array(p.value)
    except cp.error.SolverError:
        mtrx_p = None
    print("-- end solving_optimization_problem --")

    return mtrx_p, mtrx_s


def robust(
    df,
    df_base,
    num_train,
    num_bottom,
    num_agg,
    columns_agg,
    num_bootstrap,
    alpha_grid,  # sequence of alpha to be explored
    val_ratio,  # percentage of the train part devoted to validation
):
    print("== START Robust_Reconciliation ==")

    # split data
    df_trainval = df.iloc[:num_train]
    df_base_trainval = df_base.iloc[:num_train]

    n_train = int(num_train * (1.0 - val_ratio))
    df_train = df_trainval.iloc[:n_train]
    df_val = df_trainval.iloc[n_train:]
    df_base_train = df_base_trainval.iloc[:n_train]
    df_base_val = df_base_trainval.iloc[n_train:]

    best_rmse = np.inf
    best_alpha = None

    # validation
    print("---- start validation ----")
    for alpha in alpha_grid:
        print(f"-- alpha = {alpha:.2f} --")
        mtrx_p, mtrx_s = validation_alpha(
            df_train,
            df_base_train,
            n_train,
            num_bottom,
            num_agg,
            columns_agg,
            num_bootstrap,
            alpha,
        )

        # forecast
        if mtrx_p is None:
            print("No optimal solution found.")
            continue
        else:
            mtrx_sp_val = mtrx_s @ mtrx_p
            mtrx_yhat_val = df_base_val.to_numpy().T
            mtrx_ytilde_val = mtrx_sp_val @ mtrx_yhat_val

            diff = mtrx_ytilde_val.T - df_val.to_numpy()
            rmse = np.sqrt(np.mean(diff**2))
            print(f"RMSE: {rmse:.3f}")

            if rmse < best_rmse:
                best_rmse = rmse
                best_alpha = alpha
    print("---- end validation ----")

    print(f"BestAlpha: {best_alpha:.2f} (RMSE: {best_rmse:.3f})")

    # set upper and lower bounds with the best alpha
    print("---- start prediction ----")
    df_robust = robust_opt(
        df,
        df_base,
        num_train,
        num_bottom,
        num_agg,
        columns_agg,
        num_bootstrap,
        best_alpha,
    )
    print("---- end prediction ----")

    print("== END Robust_Reconciliation ==")

    return df_robust
