import numpy as np

def layernormalization(X, scale, bias, axis=-1, epsilon=1e-5):
    """
    ONNX LayerNormalization オペレータ

    レイヤー正規化を行う。

    Args:
        X: 入力テンソル
        scale: スケールパラメータ (gamma)
        bias: バイアスパラメータ (beta)
        axis: 正規化する軸 (デフォルト: -1)
        epsilon: 数値安定性のための小さな値 (デフォルト: 1e-5)

    Returns:
        Y: 正規化されたテンソル
    """
    # 平均と分散を計算
    mean = np.mean(X, axis=axis, keepdims=True)
    variance = np.var(X, axis=axis, keepdims=True)

    # 正規化
    X_normalized = (X - mean) / np.sqrt(variance + epsilon)

    # スケールとバイアスを適用
    Y = scale * X_normalized + bias

    return Y


if __name__ == "__main__":
    # テスト例
    X = np.random.randn(2, 3, 4)
    print("入力形状:", X.shape)

    # パラメータ
    scale = np.ones(4)
    bias = np.zeros(4)

    Y = layernormalization(X, scale, bias, axis=-1)
    print("出力形状:", Y.shape)

    # 正規化の確認（最後の軸に沿って平均0、分散1になっているか）
    print("\n正規化後の統計 (axis=-1):")
    print("平均:", np.mean(Y, axis=-1))
    print("分散:", np.var(Y, axis=-1))

    # 異なる軸での例
    X = np.random.randn(2, 3, 4, 5)
    scale = np.ones((4, 5))
    bias = np.zeros((4, 5))

    Y = layernormalization(X, scale, bias, axis=(2, 3))
    print("\n多次元入力:")
    print("入力形状:", X.shape)
    print("出力形状:", Y.shape)
