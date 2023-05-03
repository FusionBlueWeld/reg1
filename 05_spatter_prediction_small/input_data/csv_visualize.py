import pandas as pd
import matplotlib.pyplot as plt

# データを読み込む
df = pd.read_csv('input_data.csv', encoding="shift-jis")

# 散布図を描く
plt.scatter(df['溶接速度'], df['加工ヘッド位置'], c=df['スパッタ量'], cmap='jet')

# 軸ラベルとタイトルを設定する
plt.ylabel('Head position(mm)')
plt.xlabel('Welding speed(mm/sec)')
plt.title('Spatter amount')
plt.grid()

# カラーバーを表示する
plt.colorbar()

# 可視化を表示する
plt.show()
