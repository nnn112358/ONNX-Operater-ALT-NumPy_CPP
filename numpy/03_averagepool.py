import numpy as np

def averagepool(X, kernel_shape, strides=None, pads=None):
    """
    ONNX AveragePool オペレータ

    平均値プーリング演算を行う。

    Args:
        X: 入力テンソル (N, C, H, W)
        kernel_shape: カーネルサイズ [kH, kW]
        strides: ストライド (デフォルト: kernel_shape)
        pads: パディング [pad_top, pad_left, pad_bottom, pad_right] (デフォルト: [0, 0, 0, 0])

    Returns:
        Y: 出力テンソル
    """
    if strides is None:
        strides = kernel_shape
    if pads is None:
        pads = [0, 0, 0, 0]

    N, C, H, W = X.shape
    kH, kW = kernel_shape

    # パディング適用
    pad_top, pad_left, pad_bottom, pad_right = pads
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                      mode='constant', constant_values=0)

    # 出力サイズ計算
    out_h = (H + pad_top + pad_bottom - kH) // strides[0] + 1
    out_w = (W + pad_left + pad_right - kW) // strides[1] + 1

    # 出力テンソル初期化
    Y = np.zeros((N, C, out_h, out_w))

    # AveragePooling演算
    for n in range(N):
        for c in range(C):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * strides[0]
                    w_start = w * strides[1]

                    pool_region = X_padded[n, c, h_start:h_start+kH, w_start:w_start+kW]
                    Y[n, c, h, w] = np.mean(pool_region)

    return Y


if __name__ == "__main__":
    # テスト例
    X = np.random.randn(1, 2, 4, 4)
    print("入力形状:", X.shape)
    print("入力:\n", X[0, 0])

    Y = averagepool(X, kernel_shape=[2, 2], strides=[2, 2])
    print("\nAveragePool (2x2, stride=2):")
    print("出力形状:", Y.shape)
    print("出力:\n", Y[0, 0])
