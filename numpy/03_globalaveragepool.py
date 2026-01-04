import numpy as np

def globalaveragepool(X):
    """
    ONNX GlobalAveragePool オペレータ

    各チャネルの空間次元全体にわたる平均値プーリングを行う。

    Args:
        X: 入力テンソル (N, C, H, W) または (N, C, D, H, W)

    Returns:
        Y: 出力テンソル (N, C, 1, 1) または (N, C, 1, 1, 1)
    """
    # 空間次元（最後の2次元または3次元）に対して平均を取る
    spatial_dims = tuple(range(2, len(X.shape)))
    Y = np.mean(X, axis=spatial_dims, keepdims=True)

    return Y


if __name__ == "__main__":
    # テスト例 - 2D
    X = np.random.randn(2, 3, 4, 4)
    print("入力形状:", X.shape)
    print("入力 (チャネル0のサンプル0):\n", X[0, 0])

    Y = globalaveragepool(X)
    print("\nGlobalAveragePool:")
    print("出力形状:", Y.shape)
    print("出力値 (チャネル0のサンプル0):", Y[0, 0])

    # 検証: 手動計算
    manual_avg = np.mean(X[0, 0])
    print("手動計算での平均値:", manual_avg)

    # 3D入力の例
    X_3d = np.random.randn(1, 2, 3, 4, 4)
    Y_3d = globalaveragepool(X_3d)
    print("\n3D入力:")
    print("入力形状:", X_3d.shape)
    print("出力形状:", Y_3d.shape)
