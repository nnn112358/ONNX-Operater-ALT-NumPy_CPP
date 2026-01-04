import numpy as np

def convtranspose(X, W, B=None, strides=None, pads=None, output_padding=None):
    """
    ONNX ConvTranspose オペレータ

    転置畳み込み（逆畳み込み）演算を行う。

    Args:
        X: 入力テンソル (N, C, H, W)
        W: 重みテンソル (C, M, kH, kW)
        B: バイアス (省略可能)
        strides: ストライド (デフォルト: [1, 1])
        pads: パディング (デフォルト: [0, 0, 0, 0])
        output_padding: 出力パディング (デフォルト: [0, 0])

    Returns:
        Y: 出力テンソル
    """
    if strides is None:
        strides = [1, 1]
    if pads is None:
        pads = [0, 0, 0, 0]
    if output_padding is None:
        output_padding = [0, 0]

    N, C, H, W = X.shape
    _, M, kH, kW = W.shape

    # 出力サイズ計算
    pad_top, pad_left, pad_bottom, pad_right = pads
    out_h = (H - 1) * strides[0] - pad_top - pad_bottom + kH + output_padding[0]
    out_w = (W - 1) * strides[1] - pad_left - pad_right + kW + output_padding[1]

    # 出力テンソル初期化
    Y = np.zeros((N, M, out_h, out_w))

    # 転置畳み込み演算
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    for m in range(M):
                        for kh in range(kH):
                            for kw in range(kW):
                                h_out = h * strides[0] + kh - pad_top
                                w_out = w * strides[1] + kw - pad_left

                                if 0 <= h_out < out_h and 0 <= w_out < out_w:
                                    Y[n, m, h_out, w_out] += X[n, c, h, w] * W[c, m, kh, kw]

    # バイアス追加
    if B is not None:
        for m in range(M):
            Y[:, m, :, :] += B[m]

    return Y


if __name__ == "__main__":
    # テスト例
    X = np.random.randn(1, 2, 3, 3)  # (N, C, H, W)
    W = np.random.randn(2, 4, 3, 3)  # (C, M, kH, kW)
    B = np.random.randn(4)

    Y = convtranspose(X, W, B, strides=[2, 2], pads=[0, 0, 0, 0])
    print("入力形状:", X.shape)
    print("重み形状:", W.shape)
    print("出力形状:", Y.shape)
