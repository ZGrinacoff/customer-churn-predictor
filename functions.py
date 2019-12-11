def mean_group_df(data, field):
    mean_grouped_DF = data.groupby(field).Churn.mean().reset_index()
    return mean_grouped_DF

def count_group_df(data, field):
    count_grouped_df = data.groupby('ProbabilityCluster').field.count().reset_index()
    return count_grouped_df