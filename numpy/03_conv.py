import numpy as np

def conv(X, W, B=None, strides=None, pads=None, dilations=None, group=1):
    """
    ONNX Conv オペレータ

    畳み込み演算を行う。

    Args:
        X: 入力テンソル (N, C, H, W) または (N, C, D, H, W)
        W: 重みテンソル (M, C/group, kH, kW)
        B: バイアス (省略可能)
        strides: ストライド (デフォルト: [1, 1])
        pads: パディング [pad_top, pad_left, pad_bottom, pad_right] (デフォルト: [0, 0, 0, 0])
        dilations: 拡張率 (デフォルト: [1, 1])
        group: グループ数 (デフォルト: 1)

    Returns:
        Y: 出力テンソル
    """
    if strides is None:
        strides = [1, 1]
    if pads is None:
        pads = [0, 0, 0, 0]
    if dilations is None:
        dilations = [1, 1]

    N, C, H, W = X.shape
    M, _, kH, kW = W.shape

    # パディング適用
    pad_top, pad_left, pad_bottom, pad_right = pads
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')

    # 出力サイズ計算
    out_h = (H + pad_top + pad_bottom - dilations[0] * (kH - 1) - 1) // strides[0] + 1
    out_w = (W + pad_left + pad_right - dilations[1] * (kW - 1) - 1) // strides[1] + 1

    # 出力テンソル初期化
    Y = np.zeros((N, M, out_h, out_w))

    # 畳み込み演算
    for n in range(N):
        for m in range(M):
            for h in range(out_h):
                for w in range(out_w):
                    h_start = h * strides[0]
                    w_start = w * strides[1]

                    for c in range(C // group):
                        for kh in range(kH):
                            for kw in range(kW):
                                h_idx = h_start + kh * dilations[0]
                                w_idx = w_start + kw * dilations[1]
                                Y[n, m, h, w] += X_padded[n, c + (m // (M // group)) * (C // group), h_idx, w_idx] * W[m, c, kh, kw]

                    if B is not None:
                        Y[n, m, h, w] += B[m]

    return Y


if __name__ == "__main__":
    # テスト例
    X = np.random.randn(1, 3, 5, 5)  # (N, C, H, W)
    W = np.random.randn(2, 3, 3, 3)  # (M, C, kH, kW)
    B = np.random.randn(2)

    Y = conv(X, W, B, strides=[1, 1], pads=[0, 0, 0, 0])
    print("入力形状:", X.shape)
    print("重み形状:", W.shape)
    print("出力形状:", Y.shape)
