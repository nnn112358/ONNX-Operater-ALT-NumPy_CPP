import numpy as np

def sigmoid(X):
    """
    ONNX Sigmoid オペレータ

    シグモイド活性化関数。
    f(x) = 1 / (1 + exp(-x))

    Args:
        X: 入力テンソル

    Returns:
        Y: Sigmoidを適用した結果（0から1の範囲）
    """
    return 1 / (1 + np.exp(-X))


if __name__ == "__main__":
    # テスト例
    X = np.array([[-2, -1, 0], [1, 2, 3]])
    print("入力:")
    print(X)

    Y = sigmoid(X)
    print("\nSigmoid出力:")
    print(Y)

    # 境界値の確認
    X = np.array([-10, -5, -1, 0, 1, 5, 10])
    Y = sigmoid(X)
    print("\n詳細な入力:", X)
    print("Sigmoid出力:", Y)

    # 特性確認
    print("\nSigmoidの特性:")
    print(f"sigmoid(0) = {sigmoid(np.array(0))}")
    print(f"sigmoid(-x) + sigmoid(x) ≈ 1: {sigmoid(np.array(-2)) + sigmoid(np.array(2)):.6f}")
