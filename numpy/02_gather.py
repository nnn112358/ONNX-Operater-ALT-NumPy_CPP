import numpy as np

def gather(data, indices, axis=0):
    """
    ONNX Gather オペレータ

    指定された軸に沿って、インデックスで指定された要素を収集する。

    Args:
        data: 入力テンソル
        indices: 収集するインデックスのテンソル
        axis: 収集する軸 (デフォルト: 0)

    Returns:
        output: 収集された要素
    """
    return np.take(data, indices, axis=axis)


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("元のテンソル:")
    print(data)

    # 軸0に沿って収集
    indices = np.array([0, 2])
    result = gather(data, indices, axis=0)
    print("\nGather (indices=[0, 2], axis=0):")
    print(result)

    # 軸1に沿って収集
    indices = np.array([0, 2])
    result = gather(data, indices, axis=1)
    print("\nGather (indices=[0, 2], axis=1):")
    print(result)

    # より複雑な例
    data = np.arange(24).reshape(2, 3, 4)
    indices = np.array([0, 2])
    result = gather(data, indices, axis=2)
    print("\n3D テンソルでの Gather:")
    print("元の形状:", data.shape)
    print("結果の形状:", result.shape)
    print(result)
