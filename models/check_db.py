import pandas as pd
from sqlalchemy import create_engine

db_name = 'postgresql://tqdqfmxgrgunzx:a6d3564a45d7148a5e09817cead82db91e8b431f7521be123af79c92afd0d92c@ec2-54-91-188-254.compute-1.amazonaws.com:5432/d7saa2toh2oscb'

engine = create_engine(db_name)
conn = engine.connect()
table_name = 'DisasterResponse'

df = pd.read_sql("select * from \"DisasterResponse\"",conn)

X = df.message.values
print(X.shape)

Y = df.iloc[:,5:]

print(Y.head())

print(list(Y.columns))

print(df.head())