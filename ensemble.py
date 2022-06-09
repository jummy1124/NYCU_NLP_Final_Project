import pandas as pd
from collections import Counter

pre=[]
# cnt=[0,0,0,0]
df=pd.read_csv('./exp.csv',header=None)
sub=pd.read_csv('./pred/sub_6529.csv')
temp_df=pd.read_csv('./pred/sub_6529.csv')
for i in range(df.shape[0]):
    c = Counter(df.iloc[i,:-1])
    # print(c.most_common(5)[0],len(c.most_common()))
    if c.most_common(5)[0][1]>3 and temp_df['pred'][i]!=c.most_common(5)[0][0]:
        # cnt[1]+=1
        sub['pred'][i]=c.most_common(5)[0][0]
# print(cnt,temp_df.shape)
sub.to_csv('./ensemble.csv',index=False)