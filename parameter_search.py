import os
import pandas as pd
import random
import time
import warnings

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


# Functions to find peak values in dataframe and to shift peaks
def _find_peak_vals_idx(df_load, nlargest=1):
    day_dfs = [group[1] for group in df_load.groupby(df_load.index.date)]
    peak_vals = [day_df.nlargest(nlargest)[-1] for day_df in day_dfs]
    return [df_load.index[idx] for idx, x in enumerate(df_load) if x in peak_vals]


def _create_shifted_series(
    target_series: pd.Series, peak_vals_idx, allowed_max, highest_load_absolute
) -> pd.Series:
    shift_series = target_series.copy()

    # iterate over every load
    idx = -1
    for idx_time, val in shift_series.items():
        idx += 1
        # CASE: time steps is in blocking windows
        if idx_time in peak_vals_idx:
            # no load shifting in the last timestep:
            # last timestep:
            if idx == len(shift_series) - 1:
                print("Load shifting not possible in last timestep! Not shifting.")
                loadshift_in_last_timestep = True
                continue

            # determine current load that has to be shifted in upcoming timesteps
            # NEW ADDITIONS HERE: Use shift ratio as x % of maximum load limit
            val_shift = val - (highest_load_absolute * allowed_max)
            # If val_shift is negative, the current load is under the maximum load limit
            # Then, there is nothing to do
            if val_shift <= 0:
                continue
            # set current load:
            shift_series.iat[idx] = highest_load_absolute * allowed_max

            # shift val_shift to further timesteps, with the condition that highest heat pump load is never exceeded
            next_timestep_counter = 1
            # next timestep
            upcoming_timestep_idx = idx + 1

            # shift further until shift is empty
            while val_shift > 0:
                # check upcoming load
                next_load = shift_series.iat[upcoming_timestep_idx]
                # calculate next load if everything is shifted upon that
                possible_next_timestep_load = next_load + val_shift

                # only shift partial, since max HP load would be exceedd
                if possible_next_timestep_load > highest_load_absolute:
                    # only shift realizable
                    realizable_next_timestep_load = highest_load_absolute
                    shift_series.iat[upcoming_timestep_idx] = realizable_next_timestep_load
                    # determine remaining to be shifted
                    val_shift = possible_next_timestep_load - realizable_next_timestep_load

                    next_timestep_counter += 1
                    # jump to next timestep
                    upcoming_timestep_idx = idx + next_timestep_counter
                # no conditions harmed, shift all to next timestep
                else:
                    shift_series.iat[upcoming_timestep_idx] = possible_next_timestep_load
                    val_shift = 0

    # make sure that the sum of the shifted load is the same as the sum of the original load
    # sanity check for now
    assert round(target_series.sum(), 4) == round(shift_series.sum(), 4)

    return shift_series


def _simulation_run(
    agg_load, regular_df, params: dict, household_number: int, all_households: list, max_load_absolute: int
):
    # Initial preparations: build households

    peak_vals_idx_1 = _find_peak_vals_idx(agg_load, nlargest=1)
    shifted_df_1 = regular_df.apply(
        func=lambda x: _create_shifted_series(
            x, peak_vals_idx_1, params["initial_shift_ratio"], max_load_absolute
        )
    )

    # Based percentage of shifted households, create combined_df
    shifted_household_share_1 = params["initial_share"]
    number_shifted_households_1 = int(household_number * shifted_household_share_1)
    selected_households_shifted_1 = random.choices(
        all_households, k=number_shifted_households_1
    )

    combined_df = pd.DataFrame()
    for col in regular_df.columns:
        if col in selected_households_shifted_1:  # only replace shifted households
            combined_df.loc[:, col] = shifted_df_1.loc[:, col]
        else:
            combined_df.loc[:, col] = regular_df.loc[:, col]
    return combined_df, peak_vals_idx_1


def _evaluate_reduction(regular_df, combined_df, params=None):
    if params is None:
        params = {}
    aggregated_load_regular = regular_df.sum(axis=1)
    aggregated_load_combined = combined_df.sum(axis=1)

    max_regular = aggregated_load_regular.max()
    max_combined = aggregated_load_combined.max()
    return max_regular - max_combined


def create_random_params(config: dict):
    inital_share = random.uniform(0, config["upper_bound_initial_share"])
    initial_shift_ratio = random.uniform(0, 1)

    return {
        "initial_share": inital_share,
        "initial_shift_ratio": initial_shift_ratio,
    }


def run(repeat_cycles: int = 10, variants: int = 20, number_of_households: int = 4000, max_load_absolute: int = 9000, weeks: list[int] = None):
    if weeks is None:
        weeks = [0, 2, 4]
    # print all args in one line

    # Read in Pickle, reduce columns to households
    df_hp = pd.read_pickle(
        "./data/1920 Final Data w. Additional Features HP Hourly Agg.pkl"
    )
    df_hp_19 = df_hp[df_hp.index.year == 2019]
    df_hp_19 = df_hp_19.loc[:, [col for col in df_hp_19.columns if col.startswith("SFH")]]

    variant_results = []

    config = {"upper_bound_initial_share": 1.0}

    for _ in range(variants):
        try:
            # conduct simulation, pursue peak shaving
            params = create_random_params(config)
            print(f"Testing parameters: {params}")

            results = params.copy()

            # Create a combined dataframe of all households
            for week in weeks:
                result_week = []
                df_hp_week = df_hp_19.iloc[
                    24 * 7 * week : 24 * 7 * (week + 1)
                ]  # df of households
                for _ in range(repeat_cycles):
                    selected_households = random.choices(
                        df_hp_week.columns.tolist(), k=number_of_households
                    )
                    selected_households = [
                        f"{str(idx)} " + name
                        for idx, name in enumerate(selected_households)
                    ]

                    regular_df = pd.DataFrame()

                    dfs: list = []

                    for hh in selected_households:
                        base_name = hh.split(" ")[1]
                        dfs.append(
                            pd.DataFrame(
                                data=df_hp_week.loc[:, base_name].values,
                                columns=[hh],
                                index=df_hp_week.index,
                            )
                        )

                    regular_df = pd.concat(dfs, axis=1)

                    # Get the sum of all households, as a signal for blocking events

                    agg_load = regular_df.sum(axis=1)

                    combined_df, peak_vals = _simulation_run(
                        agg_load,
                        regular_df,
                        params,
                        number_of_households,
                        selected_households,
                        max_load_absolute
                    )

                    # calculate peak reduction
                    reduction = _evaluate_reduction(regular_df, combined_df, params)
                    result_week.append(reduction)
                results.update({f"week{str(week)}_reductions": result_week})
            variant_results.append(results)
        except KeyError as e:
            print("KeyError", e)

    res_df = pd.DataFrame(variant_results)
    week_cols = [x for x in res_df.columns if x.startswith("week")]

    for week_col in week_cols:
        res_df["MIN " + week_col] = res_df[week_col].apply(lambda x: min(x))
        res_df["AVG " + week_col] = res_df[week_col].apply(lambda x: sum(x) / len(x))

    min_cols = [x for x in res_df.columns if x.startswith("MIN ")]
    avg_cols = [x for x in res_df.columns if x.startswith("AVG ")]

    res_df["OVERALL MIN"] = res_df[min_cols].min(axis=1)
    res_df["OVERALL AVG"] = res_df[avg_cols].min(axis=1)

    # create folder if not exists
    if not os.path.exists("results"):
        os.makedirs("results")
    

    res_df.sort_values("OVERALL MIN", ascending=False).to_pickle(
        f"results/results-random-search-{time.time()}.pkl"
    )


if __name__ == "__main__":
    run()
