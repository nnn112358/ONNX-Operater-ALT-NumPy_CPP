import numpy as np

def hardswish(X):
    """
    ONNX HardSwish オペレータ

    Hard Swish活性化関数。
    f(x) = x * max(0, min(1, (x + 3) / 6))
         = x * HardSigmoid(x, alpha=1/6, beta=0.5)

    Args:
        X: 入力テンソル

    Returns:
        Y: HardSwishを適用した結果
    """
    return X * np.clip((X + 3) / 6, 0, 1)


if __name__ == "__main__":
    # テスト例
    X = np.array([[-3, -2, -1], [0, 1, 2], [3, 4, 5]])
    print("入力:")
    print(X)

    Y = hardswish(X)
    print("\nHardSwish出力:")
    print(Y)

    # より広い範囲での例
    X = np.array([-5, -3, -1, 0, 1, 3, 5, 10])
    Y = hardswish(X)
    print("\n詳細な入力:", X)
    print("HardSwish出力:", Y)

    # グラフ化用データ
    X = np.linspace(-5, 5, 100)
    Y = hardswish(X)
    print("\nHardSwishの範囲確認:")
    print(f"入力範囲: [{X.min():.2f}, {X.max():.2f}]")
    print(f"出力範囲: [{Y.min():.2f}, {Y.max():.2f}]")
