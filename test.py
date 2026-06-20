import pandas as pd

df = pd.read_csv("data/raw/steam.csv")

print(df.shape)
print(df.columns.tolist())

print(df.head())
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_year'] = df['release_date'].dt.year
df = df[df['positive_ratings'] + df['negative_ratings'] > 10]
df['review_score'] = df['positive_ratings'] / (df['positive_ratings'] + df['negative_ratings'])
df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
print(df.isnull().sum())
print(df.shape)

df.to_csv("data/processed/steam_clean.csv", index=False)
print("Saved!")