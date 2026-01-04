import numpy as np

def hardsigmoid(X, alpha=0.2, beta=0.5):
    """
    ONNX HardSigmoid オペレータ

    Hard Sigmoid活性化関数（区分線形近似）。
    f(x) = max(0, min(1, alpha * x + beta))

    Args:
        X: 入力テンソル
        alpha: 傾き (デフォルト: 0.2)
        beta: オフセット (デフォルト: 0.5)

    Returns:
        Y: HardSigmoidを適用した結果（0から1の範囲）
    """
    return np.clip(alpha * X + beta, 0, 1)


if __name__ == "__main__":
    # テスト例
    X = np.array([[-3, -2, -1], [0, 1, 2], [3, 4, 5]])
    print("入力:")
    print(X)

    Y = hardsigmoid(X)
    print("\nHardSigmoid出力 (alpha=0.2, beta=0.5):")
    print(Y)

    # 異なるパラメータでの例
    Y = hardsigmoid(X, alpha=0.5, beta=0.5)
    print("\nHardSigmoid出力 (alpha=0.5, beta=0.5):")
    print(Y)

    # 詳細な比較
    X = np.linspace(-5, 5, 11)
    Y = hardsigmoid(X, alpha=0.2, beta=0.5)
    print("\n詳細な入力:", X)
    print("HardSigmoid出力:", Y)
