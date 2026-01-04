import numpy as np

def leakyrelu(X, alpha=0.01):
    """
    ONNX LeakyRelu オペレータ

    Leaky ReLU活性化関数。
    f(x) = x if x >= 0 else alpha * x

    Args:
        X: 入力テンソル
        alpha: 負の入力に対する傾き (デフォルト: 0.01)

    Returns:
        Y: LeakyReLUを適用した結果
    """
    return np.where(X >= 0, X, alpha * X)


if __name__ == "__main__":
    # テスト例
    X = np.array([[-2, -1, 0], [1, 2, 3]])
    print("入力:")
    print(X)

    Y = leakyrelu(X, alpha=0.01)
    print("\nLeakyReLU出力 (alpha=0.01):")
    print(Y)

    Y = leakyrelu(X, alpha=0.1)
    print("\nLeakyReLU出力 (alpha=0.1):")
    print(Y)

    # グラフ化できるデータ
    X = np.linspace(-3, 3, 100)
    Y = leakyrelu(X, alpha=0.01)
    print("\nLeakyReLU (alpha=0.01) の範囲確認:")
    print(f"入力範囲: [{X.min():.2f}, {X.max():.2f}]")
    print(f"出力範囲: [{Y.min():.2f}, {Y.max():.2f}]")
