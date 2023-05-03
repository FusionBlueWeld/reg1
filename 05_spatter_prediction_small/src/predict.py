import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

root_path = "C:/Users/FUTOSHI/Desktop/ChatGPT_test/05_spatter_prediction_small/"
input_data_path = os.path.join(root_path, "input_data/input_data.csv")
output_model_path = os.path.join(root_path, "output_data/trained_welding_model.h5")

# 入力データの読み込み
data = pd.read_csv(input_data_path, encoding="shift-jis")
input_data = data[['加工ヘッド位置', '溶接速度']].values
target_data = data['スパッタ量'].values

# 入力データの正規化
scaler = MinMaxScaler()
input_data_normalized = scaler.fit_transform(input_data)

# 学習済みモデルの読み込み
model = load_model(output_model_path)

# 予測を実行
x_positions = np.linspace(-6.0, 6.0, 121)
welding_speeds = np.linspace(30, 600, 571)
X, Y = np.meshgrid(x_positions, welding_speeds)
input_points = np.array([X.flatten(), Y.flatten()]).T
input_points_normalized = scaler.transform(input_points)
predictions = model.predict(input_points_normalized).reshape(X.shape)

# 予測マップのプロット
plt.figure(figsize=(10, 7))
plt.pcolormesh(Y, X, predictions, cmap='jet', alpha=0.5, shading='auto')
plt.scatter(input_data[:, 1], input_data[:, 0], c=target_data, cmap='jet', edgecolors='k', s=50)
plt.colorbar(label='Spatter amount')
plt.ylabel('Head position [mm]')
plt.xlabel('Welding speed [mm/sec]')
plt.title('Spatter prediction map (Power: 1900W, Thickness: 3.0mm)')
plt.xlim(20,610)
plt.ylim(-6.3,6.3)
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(root_path, "output_data/prediction_map.png"))



# # ルートパスの設定
# root_path = "C:/Users/FUTOSHI/Desktop/ChatGPT_test/05_spatter_prediction_small/"
# output_figure_path = os.path.join(root_path, "output_data/spatter_prediction_map.png")

# # 学習済モデルの読み込み
# model_path = os.path.join(root_path, "output_data/trained_welding_model.h5")
# model = load_model(model_path)

# # 入力データの範囲
# head_position = np.linspace(-6.0, 6.0, 100)
# welding_speed = np.linspace(30.0, 600.0, 100)

# # 空の2次元配列を作成
# predictions = np.zeros((len(head_position), len(welding_speed)))

# # 入力データを生成
# input_data = np.array([[pos, speed] for pos in head_position for speed in welding_speed])

# # 入力データを正規化
# scaler = MinMaxScaler()
# input_data_normalized = scaler.fit_transform(input_data)

# # 予測を実行
# predictions_flat = model.predict(input_data_normalized)

# # 予測結果を二次元配列に変換
# predictions = predictions_flat.reshape(len(head_position), len(welding_speed))



# # 2次元マップの表示
# plt.figure(figsize=(10, 7))
# plt.imshow(predictions, cmap='jet', origin='lower', extent=(-6.0, 6.0, 30.0, 600.0), aspect='auto')
# plt.xlabel('Welding Speed (mm/sec)')
# plt.ylabel('Head Position (mm)')
# plt.title('Spatter Prediction Map')
# plt.colorbar(label='Spatter Amount')
# plt.tight_layout()

# # グラフの保存
# plt.savefig(output_figure_path)