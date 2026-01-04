import numpy as np

def elu(X, alpha=1.0):
    """
    ONNX Elu オペレータ

    ELU（Exponential Linear Unit）活性化関数。
    f(x) = x if x >= 0 else alpha * (exp(x) - 1)

    Args:
        X: 入力テンソル
        alpha: 負の入力に対するスケール (デフォルト: 1.0)

    Returns:
        Y: ELUを適用した結果
    """
    return np.where(X >= 0, X, alpha * (np.exp(X) - 1))


if __name__ == "__main__":
    # テスト例
    X = np.array([[-2, -1, 0], [1, 2, 3]])
    print("入力:")
    print(X)

    Y = elu(X, alpha=1.0)
    print("\nELU出力 (alpha=1.0):")
    print(Y)

    Y = elu(X, alpha=0.5)
    print("\nELU出力 (alpha=0.5):")
    print(Y)

    # より詳細な例
    X = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3])
    Y = elu(X, alpha=1.0)
    print("\n詳細な入力:", X)
    print("ELU出力:", Y)
