import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def lstm(X, W, R, B=None, initial_h=None, initial_c=None):
    """
    ONNX LSTM オペレータ

    LSTMセルの順伝播を行う。

    Args:
        X: 入力テンソル (seq_length, batch_size, input_size)
        W: 入力重み (num_directions, 4*hidden_size, input_size)
        R: リカレント重み (num_directions, 4*hidden_size, hidden_size)
        B: バイアス (num_directions, 8*hidden_size) - [Wb, Rb] を連結
        initial_h: 初期隠れ状態 (num_directions, batch_size, hidden_size)
        initial_c: 初期セル状態 (num_directions, batch_size, hidden_size)

    Returns:
        Y: 出力テンソル (seq_length, num_directions, batch_size, hidden_size)
        Y_h: 最終隠れ状態
        Y_c: 最終セル状態
    """
    seq_length, batch_size, input_size = X.shape
    num_directions, four_hidden, _ = W.shape
    hidden_size = four_hidden // 4

    # 初期状態の設定
    if initial_h is None:
        h = np.zeros((num_directions, batch_size, hidden_size))
    else:
        h = initial_h.copy()

    if initial_c is None:
        c = np.zeros((num_directions, batch_size, hidden_size))
    else:
        c = initial_c.copy()

    # バイアスの設定
    if B is None:
        Wb = np.zeros((num_directions, 4 * hidden_size))
        Rb = np.zeros((num_directions, 4 * hidden_size))
    else:
        Wb = B[:, :4*hidden_size]
        Rb = B[:, 4*hidden_size:]

    Y = np.zeros((seq_length, num_directions, batch_size, hidden_size))

    # 各方向について処理（通常は1方向）
    for d in range(num_directions):
        h_d = h[d]
        c_d = c[d]

        for t in range(seq_length):
            x_t = X[t]  # (batch_size, input_size)

            # ゲート計算
            gates = np.dot(x_t, W[d].T) + np.dot(h_d, R[d].T) + Wb[d] + Rb[d]

            # 4つのゲートに分割
            i = sigmoid(gates[:, :hidden_size])                        # input gate
            f = sigmoid(gates[:, hidden_size:2*hidden_size])           # forget gate
            g = tanh(gates[:, 2*hidden_size:3*hidden_size])            # cell gate
            o = sigmoid(gates[:, 3*hidden_size:])                      # output gate

            # セル状態と隠れ状態の更新
            c_d = f * c_d + i * g
            h_d = o * tanh(c_d)

            Y[t, d] = h_d

        # 最終状態を保存
        h[d] = h_d
        c[d] = c_d

    return Y, h, c


if __name__ == "__main__":
    # テスト例
    seq_length = 3
    batch_size = 2
    input_size = 4
    hidden_size = 5

    X = np.random.randn(seq_length, batch_size, input_size)
    W = np.random.randn(1, 4*hidden_size, input_size)
    R = np.random.randn(1, 4*hidden_size, hidden_size)
    B = np.random.randn(1, 8*hidden_size)

    Y, Y_h, Y_c = lstm(X, W, R, B)

    print("入力形状:", X.shape)
    print("出力形状:", Y.shape)
    print("最終隠れ状態形状:", Y_h.shape)
    print("最終セル状態形状:", Y_c.shape)
