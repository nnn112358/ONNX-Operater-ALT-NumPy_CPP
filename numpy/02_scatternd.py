import numpy as np

def scatternd(data, indices, updates):
    """
    ONNX ScatterND オペレータ

    インデックスで指定された位置に更新値を散布する。

    Args:
        data: 入力テンソル (ベーステンソル)
        indices: 更新する位置のインデックス
        updates: 更新値

    Returns:
        output: 更新されたテンソル
    """
    output = np.copy(data)

    # indicesの形状に応じて処理
    if indices.ndim == 1:
        indices = indices.reshape(-1, 1)

    # 各インデックスに対して更新を適用
    for i in range(indices.shape[0]):
        idx = tuple(indices[i])
        output[idx] = updates[i]

    return output


if __name__ == "__main__":
    # テスト例1: 1D配列
    data = np.zeros(8)
    indices = np.array([[1], [3], [5]])
    updates = np.array([10, 20, 30])

    result = scatternd(data, indices, updates)
    print("1D例:")
    print("元のデータ:", data)
    print("結果:", result)

    # テスト例2: 2D配列
    data = np.zeros((4, 4))
    indices = np.array([[0, 1], [1, 2], [2, 3]])
    updates = np.array([1, 2, 3])

    result = scatternd(data, indices, updates)
    print("\n2D例:")
    print("結果:")
    print(result)

    # テスト例3: 既存データの更新
    data = np.ones((3, 3)) * 5
    indices = np.array([[0, 0], [1, 1], [2, 2]])
    updates = np.array([10, 20, 30])

    result = scatternd(data, indices, updates)
    print("\n対角要素の更新:")
    print(result)
