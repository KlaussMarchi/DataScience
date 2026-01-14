import os, ast
import pandas as pd

if not os.path.exists('files'):
    os.makedirs('files')


df = pd.read_csv('files/raw.csv')
print(df)
print()
data = []

while True:
    values = ast.literal_eval(input('\ncole os dados: ').strip())
    values = [int(val) for val in values]

    print(f"\nrecebido {len(values)} dados: {values}\n")

    if values is None:
        continue

    result = int(input('alcool: ').strip().split(' ')[-1])
    info   = {'data': values, 'alcohol': bool(result)}
    data.append(info)

    if input('deseja continuar? ') == 'n':
        print('encerrando')
        break

data = pd.DataFrame(data)
df   = pd.concat([df, data], axis=0)
df.to_csv('files/raw.csv', index=False)