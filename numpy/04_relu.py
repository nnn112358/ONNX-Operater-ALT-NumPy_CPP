import numpy as np

def relu(X):
    """
    ONNX Relu オペレータ

    ReLU（Rectified Linear Unit）活性化関数。
    f(x) = max(0, x)

    Args:
        X: 入力テンソル

    Returns:
        Y: ReLUを適用した結果
    """
    return np.maximum(0, X)


if __name__ == "__main__":
    # テスト例
    X = np.array([[-2, -1, 0], [1, 2, 3]])
    print("入力:")
    print(X)

    Y = relu(X)
    print("\nReLU出力:")
    print(Y)

    # 浮動小数点数の例
    X = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    print("\n浮動小数点入力:", X)
    print("ReLU出力:", relu(X))
