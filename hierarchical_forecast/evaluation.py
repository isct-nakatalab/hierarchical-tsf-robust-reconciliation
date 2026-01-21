import collections
import pandas as pd


def evaluation(
    df,
    df_base,
    df_bu,
    df_td,
    df_gls,
    df_mint,
    df_robust,
    num_test,
):
    print("== START Evaluation ==")

    # calculate MAE
    mae = dict()
    df_base_diff = df_base - df
    df_bu_diff = df_bu - df
    df_td_diff = df_td - df
    df_gls_diff = df_gls - df
    df_mint_diff = df_mint - df
    df_robust_diff = df_robust - df
    mae["base"] = df_base_diff.iloc[-num_test:].abs().mean()
    mae["bu"] = df_bu_diff.iloc[-num_test:].abs().mean()
    mae["td"] = df_td_diff.iloc[-num_test:].abs().mean()
    mae["gls"] = df_gls_diff.iloc[-num_test:].abs().mean()
    mae["mint"] = df_mint_diff.iloc[-num_test:].abs().mean()
    mae["robust"] = df_robust_diff.iloc[-num_test:].abs().mean()
    mae = pd.DataFrame(mae)

    # calculate RMSE
    rmse = dict()
    df_base_diff_sq = df_base_diff**2
    df_bu_diff_sq = df_bu_diff**2
    df_td_diff_sq = df_td_diff**2
    df_gls_diff_sq = df_gls_diff**2
    df_mint_diff_sq = df_mint_diff**2
    df_robust_diff_sq = df_robust_diff**2
    rmse["base"] = df_base_diff_sq.iloc[-num_test:].mean() ** 0.5
    rmse["bu"] = df_bu_diff_sq.iloc[-num_test:].mean() ** 0.5
    rmse["td"] = df_td_diff_sq.iloc[-num_test:].mean() ** 0.5
    rmse["gls"] = df_gls_diff_sq.iloc[-num_test:].mean() ** 0.5
    rmse["mint"] = df_mint_diff_sq.iloc[-num_test:].mean() ** 0.5
    rmse["robust"] = df_robust_diff_sq.iloc[-num_test:].mean() ** 0.5
    rmse = pd.DataFrame(rmse)

    # print summary
    print(
        pd.concat(
            [mae.mean(), rmse.mean(), mae.std(), rmse.std()],
            axis=1,
            keys=[
                "mae_ave",
                "rmse_ave",
                "mae_std",
                "rmse_std",
            ],
        )
    )
    print("-----")

    # count the best method in each series
    mae_best = []
    for _, mae_i in mae.iterrows():
        mae_best += [mae_i.idxmin()]
    print(f"MAE_BestMethod: {collections.Counter(mae_best)}")
    rmse_best = []
    for _, rmse_i in rmse.iterrows():
        rmse_best += [rmse_i.idxmin()]
    print(f"RMSE_BestMethod: {collections.Counter(rmse_best)}")

    print("== END Evaluation ==")

    # return MAE, RMSE
    return mae, rmse
