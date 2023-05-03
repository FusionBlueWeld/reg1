import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import SGD

# GPU を使用しない設定
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ニューラルネットのパラメータ初期値を固定
tf.random.set_seed(42)

# ルートパスの設定
root_path = "C:/Users/FUTOSHI/Desktop/ChatGPT_test/05_spatter_prediction_small/"

# 学習モデルの設計（いろいろ試して一番良かったやつ）
def create_welding_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(2,), kernel_initializer='he_uniform', bias_initializer='zeros'),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_initializer='he_uniform', bias_initializer='zeros'),
        Dropout(0.2),
        Dense(1, activation='sigmoid', kernel_initializer='he_uniform', bias_initializer='zeros')
    ])

    optimizer = SGD(lr=0.1, momentum=0.9)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model


# モデルの学習
def train_welding_model(model, input_data, target_data, epochs=100):
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(input_data, target_data, epochs=epochs, validation_split=0.2, callbacks=[early_stop], verbose=0)
    return history

# MSE と R^2 スコアの推移をグラフ化
def plot_scores(history):
    mse_train = history.history['mse']
    mse_val = history.history['val_mse']
    r2_train = [r2_score(y_train, model.predict(X_train)) for _ in history.epoch]
    r2_val = [r2_score(y_test, model.predict(X_test)) for _ in history.epoch]

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE')

    ax1.plot(mse_train, label='Train MSE', color='red', linestyle='-')
    ax1.plot(mse_val, label='Validation MSE', color='red', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.grid(True)


    ax2 = ax1.twinx()
    ax2.set_ylabel('R^2 Score')

    ax2.plot(r2_train, label='Train R^2', color='blue', linestyle='-')
    ax2.plot(r2_val, label='Validation R^2', color='blue', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='blue')

    fig.tight_layout()
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, bbox_to_anchor=(1.5, 1))

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)

    output_graph_path = os.path.join(root_path, "output_data/mse_r2_scores.png")
    plt.savefig(output_graph_path, bbox_inches='tight')

# 評価結果を表示する関数
def print_evaluation_results(history, X_train, y_train, X_test, y_test, model):
    train_mse = history.history['mse'][-1]
    val_mse = history.history['val_mse'][-1]
    train_r2 = r2_score(y_train, model.predict(X_train))
    val_r2 = r2_score(y_test, model.predict(X_test))

    print(f"Epochs: {len(history.epoch)}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Train R^2: {train_r2:.4f}")
    print(f"Validation R^2: {val_r2:.4f}")


if __name__ == "__main__":
    # 入力データと出力データのパスを設定
    input_data_path = os.path.join(root_path, "input_data/input_data.csv")
    output_model_path = os.path.join(root_path, "output_data/trained_welding_model.h5")

    # 入力データの読み込み
    data = pd.read_csv(input_data_path, encoding="shift-jis")
    input_data = data[['加工ヘッド位置', '溶接速度']].values
    target_data = data['スパッタ量'].values

    # 入力データの正規化
    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data)

    # 学習データとテストデータに分離
    X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=0.2, random_state=42)

    # モデルの作成と学習の実行
    model = create_welding_model()
    history = train_welding_model(model, X_train, y_train)

    # 学習済モデルの保存
    model.save(output_model_path)

    # MSE と R^2 スコアの推移をグラフ化して保存
    plot_scores(history)

    # 評価結果を表示
    print_evaluation_results(history, X_train, y_train, X_test, y_test, model)