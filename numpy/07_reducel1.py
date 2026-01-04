import numpy as np

def reducel1(data, axes=None, keepdims=True):
    """
    ONNX ReduceL1 オペレータ

    指定された軸に沿ってL1ノルムを計算する。
    L1 = sum(|x|)

    Args:
        data: 入力テンソル
        axes: 削減する軸のリスト (省略時は全軸)
        keepdims: 次元を保持するか (デフォルト: True)

    Returns:
        reduced: L1ノルム
    """
    if axes is not None:
        axes = tuple(axes) if isinstance(axes, list) else axes
    return np.sum(np.abs(data), axis=axes, keepdims=keepdims)


if __name__ == "__main__":
    # テスト例
    data = np.array([[1, -2, 3], [-4, 5, -6]])
    print("入力:")
    print(data)

    # 全要素のL1ノルム
    result = reducel1(data, axes=None)
    print("\nReduceL1 (全軸):", result)
    print("検証: |1| + |-2| + |3| + |-4| + |5| + |-6| =", 1 + 2 + 3 + 4 + 5 + 6)

    # 軸0に沿ったL1ノルム
    result = reducel1(data, axes=[0], keepdims=True)
    print("\nReduceL1 (axis=0, keepdims=True):")
    print(result)

    # 軸1に沿ったL1ノルム
    result = reducel1(data, axes=[1], keepdims=True)
    print("\nReduceL1 (axis=1, keepdims=True):")
    print(result)
    print("検証: [|1| + |-2| + |3|, |-4| + |5| + |-6|] =", [1 + 2 + 3, 4 + 5 + 6])

    # keepdims=False
    result = reducel1(data, axes=[1], keepdims=False)
    print("\nReduceL1 (axis=1, keepdims=False):")
    print(result)
