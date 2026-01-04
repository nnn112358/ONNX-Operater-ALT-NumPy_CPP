import numpy as np

def swish(X):
    """
    ONNX Swish オペレータ

    Swish活性化関数（SiLUとも呼ばれる）。
    f(x) = x * sigmoid(x) = x / (1 + exp(-x))

    Args:
        X: 入力テンソル

    Returns:
        Y: Swishを適用した結果
    """
    return X / (1 + np.exp(-X))


if __name__ == "__main__":
    # テスト例
    X = np.array([[-2, -1, 0], [1, 2, 3]])
    print("入力:")
    print(X)

    Y = swish(X)
    print("\nSwish出力:")
    print(Y)

    # より広い範囲での例
    X = np.array([-5, -3, -1, 0, 1, 3, 5])
    Y = swish(X)
    print("\n詳細な入力:", X)
    print("Swish出力:", Y)

    # Swishの特性確認
    X = np.linspace(-5, 5, 100)
    Y = swish(X)
    print("\nSwishの範囲確認:")
    print(f"入力範囲: [{X.min():.2f}, {X.max():.2f}]")
    print(f"出力範囲: [{Y.min():.2f}, {Y.max():.2f}]")
