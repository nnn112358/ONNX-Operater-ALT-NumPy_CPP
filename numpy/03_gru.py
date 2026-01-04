import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def gru(X, W, R, B=None, initial_h=None):
    """
    ONNX GRU オペレータ

    GRUセルの順伝播を行う。

    Args:
        X: 入力テンソル (seq_length, batch_size, input_size)
        W: 入力重み (num_directions, 3*hidden_size, input_size)
        R: リカレント重み (num_directions, 3*hidden_size, hidden_size)
        B: バイアス (num_directions, 6*hidden_size) - [Wb, Rb] を連結
        initial_h: 初期隠れ状態 (num_directions, batch_size, hidden_size)

    Returns:
        Y: 出力テンソル (seq_length, num_directions, batch_size, hidden_size)
        Y_h: 最終隠れ状態
    """
    seq_length, batch_size, input_size = X.shape
    num_directions, three_hidden, _ = W.shape
    hidden_size = three_hidden // 3

    # 初期状態の設定
    if initial_h is None:
        h = np.zeros((num_directions, batch_size, hidden_size))
    else:
        h = initial_h.copy()

    # バイアスの設定
    if B is None:
        Wb = np.zeros((num_directions, 3 * hidden_size))
        Rb = np.zeros((num_directions, 3 * hidden_size))
    else:
        Wb = B[:, :3*hidden_size]
        Rb = B[:, 3*hidden_size:]

    Y = np.zeros((seq_length, num_directions, batch_size, hidden_size))

    # 各方向について処理（通常は1方向）
    for d in range(num_directions):
        h_d = h[d]

        for t in range(seq_length):
            x_t = X[t]  # (batch_size, input_size)

            # ゲート計算（update gate と reset gate）
            gates_input = np.dot(x_t, W[d].T) + Wb[d]
            gates_hidden = np.dot(h_d, R[d].T) + Rb[d]

            # update gate (z) と reset gate (r)
            z = sigmoid(gates_input[:, :hidden_size] + gates_hidden[:, :hidden_size])
            r = sigmoid(gates_input[:, hidden_size:2*hidden_size] + gates_hidden[:, hidden_size:2*hidden_size])

            # 候補隠れ状態
            h_tilde = tanh(gates_input[:, 2*hidden_size:] + r * gates_hidden[:, 2*hidden_size:])

            # 隠れ状態の更新
            h_d = (1 - z) * h_tilde + z * h_d

            Y[t, d] = h_d

        # 最終状態を保存
        h[d] = h_d

    return Y, h


if __name__ == "__main__":
    # テスト例
    seq_length = 3
    batch_size = 2
    input_size = 4
    hidden_size = 5

    X = np.random.randn(seq_length, batch_size, input_size)
    W = np.random.randn(1, 3*hidden_size, input_size)
    R = np.random.randn(1, 3*hidden_size, hidden_size)
    B = np.random.randn(1, 6*hidden_size)

    Y, Y_h = gru(X, W, R, B)

    print("入力形状:", X.shape)
    print("出力形状:", Y.shape)
    print("最終隠れ状態形状:", Y_h.shape)
