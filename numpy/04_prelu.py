import numpy as np

def prelu(X, slope):
    """
    ONNX PRelu オペレータ

    PReLU（Parametric ReLU）活性化関数。
    f(x) = x if x >= 0 else slope * x
    slopeは学習可能なパラメータ。

    Args:
        X: 入力テンソル
        slope: 負の入力に対する傾きパラメータ（チャネルごとまたは全体で共有）

    Returns:
        Y: PReL uを適用した結果
    """
    return np.where(X >= 0, X, slope * X)


if __name__ == "__main__":
    # テスト例1: スカラーのslope
    X = np.array([[-2, -1, 0], [1, 2, 3]])
    slope = 0.25
    print("入力:")
    print(X)
    print(f"\nSlope (スカラー): {slope}")

    Y = prelu(X, slope)
    print("PReLU出力:")
    print(Y)

    # テスト例2: チャネルごとのslope
    X = np.random.randn(2, 3, 4, 4)  # (N, C, H, W)
    slope = np.array([0.1, 0.2, 0.3]).reshape(1, 3, 1, 1)
    print("\n\n多次元入力形状:", X.shape)
    print("Slope形状:", slope.shape)

    Y = prelu(X, slope)
    print("PReLU出力形状:", Y.shape)

    # テスト例3: 異なるslopeの比較
    X = np.array([-2, -1, 0, 1, 2])
    for s in [0.01, 0.1, 0.25, 0.5]:
        Y = prelu(X, s)
        print(f"\nSlope={s}: {Y}")
