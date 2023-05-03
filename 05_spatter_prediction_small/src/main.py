import os
import regression
import predict

# io関係のパス設定
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
input_path = root_path + "\\input_data"
output_path = root_path + "\\output_data"
model_path = output_path + "\\trained_welding_model.h5"

# スパッタマップの変数設定
laser_power = "var_y"
head_position = 3
welding_speed = "var_x"
work_thickness = 0.5

def main(learning = False):

    if learning == True:
        # 学習を実行する
        regression.train_regression_model(input_path, output_path)
    else:
        print("学習は実行しません")

    # 予測を実行
    # predict.make_prediction(model_path, output_path, laser_power, head_position, welding_speed, work_thickness)

if __name__ == "__main__":
    main(True)