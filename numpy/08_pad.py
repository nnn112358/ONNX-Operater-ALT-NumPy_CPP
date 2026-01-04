import numpy as np

def pad(data, pads, mode='constant', constant_value=0):
    """
    ONNX Pad オペレータ

    テンソルにパディングを追加する。

    Args:
        data: 入力テンソル
        pads: パディング量 [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
        mode: パディングモード ('constant', 'reflect', 'edge')
        constant_value: constantモード時の値 (デフォルト: 0)

    Returns:
        padded: パディングされたテンソル
    """
    ndim = len(data.shape)

    # padsを[(begin, end), ...]の形式に変換
    pad_width = []
    for i in range(ndim):
        begin = pads[i]
        end = pads[i + ndim]
        pad_width.append((begin, end))

    # モードの変換
    if mode == 'constant':
        return np.pad(data, pad_width, mode='constant', constant_values=constant_value)
    elif mode == 'reflect':
        return np.pad(data, pad_width, mode='reflect')
    elif mode == 'edge':
        return np.pad(data, pad_width, mode='edge')
    else:
        return np.pad(data, pad_width, mode='constant', constant_values=constant_value)


if __name__ == "__main__":
    # テスト例1: 2D配列のパディング
    data = np.array([[1, 2], [3, 4]])
    print("入力:")
    print(data)

    # constantモード
    pads = [1, 1, 1, 1]  # 上下左右に1ずつ
    result = pad(data, pads, mode='constant', constant_value=0)
    print("\nPad (constant, value=0):")
    print(result)

    # reflectモード
    result = pad(data, pads, mode='reflect')
    print("\nPad (reflect):")
    print(result)

    # edgeモード
    result = pad(data, pads, mode='edge')
    print("\nPad (edge):")
    print(result)

    # 非対称パディング
    data = np.array([[1, 2, 3], [4, 5, 6]])
    pads = [0, 1, 2, 1]  # 上0, 左1, 下2, 右1
    result = pad(data, pads, mode='constant', constant_value=9)
    print("\n非対称パディング:")
    print("入力形状:", data.shape)
    print("出力形状:", result.shape)
    print(result)
