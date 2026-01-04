import numpy as np

def tanh(X):
    """
    ONNX Tanh オペレータ

    ハイパボリックタンジェント活性化関数。
    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Args:
        X: 入力テンソル

    Returns:
        Y: Tanhを適用した結果（-1から1の範囲）
    """
    return np.tanh(X)


if __name__ == "__main__":
    # テスト例
    X = np.array([[-2, -1, 0], [1, 2, 3]])
    print("入力:")
    print(X)

    Y = tanh(X)
    print("\nTanh出力:")
    print(Y)

    # より広い範囲での例
    X = np.array([-5, -3, -1, 0, 1, 3, 5])
    Y = tanh(X)
    print("\n詳細な入力:", X)
    print("Tanh出力:", Y)

    # 特性確認
    print("\nTanhの特性:")
    print(f"tanh(0) = {tanh(np.array(0))}")
    print(f"tanh(-x) = -tanh(x): tanh(-2) = {tanh(np.array(-2)):.4f}, -tanh(2) = {-tanh(np.array(2)):.4f}")

    # 範囲確認
    X = np.linspace(-5, 5, 100)
    Y = tanh(X)
    print("\nTanhの範囲確認:")
    print(f"入力範囲: [{X.min():.2f}, {X.max():.2f}]")
    print(f"出力範囲: [{Y.min():.4f}, {Y.max():.4f}]")
