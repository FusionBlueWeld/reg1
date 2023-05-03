import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 学習結果のグラフ出力
def plot_graph(history, output_path):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(history.history['loss'], label='Training Loss', color='b')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='b', linestyle='dashed')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_ylim(bottom=0, top=0.5)

    ax2.plot(history.history['mae'], label='Training MAE', color='r')
    ax2.plot(history.history['val_mae'], label='Validation MAE', color='r', linestyle='dashed')
    ax2.set_ylabel('MAE')
    ax2.set_ylim(bottom=0, top=0.5)

    # 凡例とグリッド線の設定
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.9))

    ax1.grid(True)
    ax2.grid(True)
    plt.title('Loss and MAE')

    # グラフを枠内に収める
    plt.tight_layout()

    # グラフ画像を同じフォルダに保存
    graph_name = output_path + "\\loss_and_mae.png"
    plt.savefig(graph_name)

# モデルの出力評価
def plot_scatter(y_true, y_pred, output_path):
    plt.figure()
    plt.scatter(y_true, y_pred)
    plt.plot([0, 1], [0, 1], 'r')
    plt.xlabel('True Spatter Amount')
    plt.ylabel('Predicted Spatter Amount')
    plt.title('Scatter Plot: True vs Predicted Spatter Amount')
    plt.savefig(output_path + "/scatter_plot.png")




# スパッタマップのグラフ出力、関連の関数群

# 入力値を取得する関数
def get_input_params(laser_power, head_position, welding_speed, work_thickness):
    return {
        'laser_power': laser_power,
        'head_position': head_position,
        'welding_speed': welding_speed,
        'work_thickness': work_thickness
    }

# 変数パラメータの範囲と分解能を取得する関数
def get_param_ranges():
    return {
        'laser_power': np.arange(200, 2001, 10),
        'head_position': np.arange(-6.0, 6.1, 0.1),
        'welding_speed': np.arange(10, 651, 10),
        'work_thickness': np.arange(0.5, 6.1, 0.1)
    }

# パラメータの単位を取得する関数
def get_unit_dict():
    return {
        'laser_power': 'W',
        'head_position': 'mm',
        'welding_speed': 'mm/sec',
        'work_thickness': 'mm'
    }

# 固定値と変数の設定を取得する関数
def get_fixed_and_variable_params(params):
    fixed_params = {k: v for k, v in params.items() if v != 'var_x' and v != 'var_y'}
    variable_params = {k: v for k, v in params.items() if v == 'var_x' or v == 'var_y'}
    return fixed_params, variable_params

# 入力パラメータを準備する関数
def prepare_input_params(param_ranges, fixed_params, variable_params):
    x_var, y_var = [k for k, v in variable_params.items() if v == 'var_x'][0], [k for k, v in variable_params.items() if v == 'var_y'][0]
    x_range, y_range = param_ranges[x_var], param_ranges[y_var]

    grid_x, grid_y = np.meshgrid(x_range, y_range)
    grid_x_flat, grid_y_flat = grid_x.flatten(), grid_y.flatten()

    input_params = np.array([[fixed_params.get('laser_power', x), 
                                fixed_params.get('head_position', y), 
                                fixed_params.get('welding_speed', x), 
                                fixed_params.get('work_thickness', y)] 
                                for x, y in zip(grid_x_flat, grid_y_flat)])
    return x_var, y_var, x_range, y_range, grid_x, grid_y, input_params

# スパッタマップをプロットする関数
def plot_spatter_map(output_path, predicted_spatter, x_var, y_var, x_range, y_range, fixed_params, unit_dict):
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    plt.figure(figsize=(10, 8))
    c = plt.pcolormesh(grid_x, grid_y, predicted_spatter, cmap='jet', shading='auto', vmin=0)

    plt.colorbar(c, label='Spatter Amount')
    x_label = x_var.replace('_', ' ')
    y_label = y_var.replace('_', ' ')
    plt.xlabel(f'{x_label} ({unit_dict[x_var]})')
    plt.ylabel(f'{y_label} ({unit_dict[y_var]})')
    fixed_params_str = ', '.join([f'{k}: {v} {unit_dict[k]}' for k, v in fixed_params.items()])
    plt.title(f'Spatter Map / {fixed_params_str}')

    plt.grid(True)
    plt.xlim(x_range.min(), x_range.max())
    plt.ylim(y_range.min(), y_range.max())

    fixed_params_str = ', '.join([f'{k}: {v} {unit_dict[k]}' for k, v in fixed_params.items()])
    graph_title = f'Spatter Map / {fixed_params_str}'

    # 使用できない文字を置き換える
    filename = graph_title.replace('/', '-').replace(':', '').replace(',', '-')

    # 画像ファイルの出力
    plt.savefig(output_path + f'\\{filename}.png')

