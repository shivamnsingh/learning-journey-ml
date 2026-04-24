def show_sum(df):
    sum_df = pd.DataFrame(index=list(df))
    
    sum_df['Dtype']    = df.dtypes
    sum_df['Count']    = df.count()
    sum_df['#Unique']  = df.nunique()
    sum_df['#Missing'] = df.isnull().sum()
    sum_df['%Missing'] = (df.isnull().sum() / len(df) * 100).round(2)
    
    return sum_df
