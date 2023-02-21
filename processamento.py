import pandas as pd

path = '/content/dataset/yelp_academic_dataset_business.csv'
df = pd.read_csv(path)

dic_categories = {}
for i in range(df.shape[0]):
  l_cate = str(df["categories"][i]).split(',')
  for cat in l_cate:
    cat = cat.lstrip()
    if cat in dic_categories:
      dic_categories[cat] = dic_categories[cat]+1
    else:
      dic_categories[cat] = 1

dic_categories_o = dict(sorted(dic_categories.items(), key=lambda item: item[1],reverse=True))

cats = []
for i in range(df.shape[0]):
  dic_i = {}
  l = str(df["categories"][i]).split(',')
  for i in l:
    i = i.lstrip()
    dic_i[i] = dic_categories_o[i]

  dic_o = dict(sorted(dic_i.items(), key=lambda item: item[1],reverse=True))
  itens = list(dic_o.items())
  primeiros_tres_itens = itens[:3]
  if len(primeiros_tres_itens) == 3:
    cat_1 = primeiros_tres_itens[0][0]
    cat_2 = primeiros_tres_itens[1][0]
    cat_3 = primeiros_tres_itens[2][0]
    text = f'{cat_1},{cat_2} and {cat_3}'
  elif len(primeiros_tres_itens) == 2:
    cat_1 = primeiros_tres_itens[0][0]
    cat_2 = primeiros_tres_itens[1][0]
    text = f'{cat_1} and {cat_2}'
  else:
    cat_1 = primeiros_tres_itens[0][0]
    text = f'{cat_1}'
  cats.append(text)

df['cate'] = cats

df['is_open'] = df['is_open'].replace([0], 'closed').replace([1], 'opened')

for index, row in df.iterrows():
    if row['stars'] > 3.5:
        df['stars'][index] = 'good'

    elif row['stars'] < 2.5:
        df['stars'][index] = 'bad'

    else:
        df['stars'][index] = 'neutral'

def concat_columns(row):
    return f'{row["name"]} {str(row["cate"])} {row["stars"]} {row["attributes"]}'

df['features'] = df.apply(concat_columns, axis=1)
df1 = df[['name','city','cate','stars','is_open','features','business_id']]
df1.to_csv('/content/dataset/df_categories.csv', index=False)