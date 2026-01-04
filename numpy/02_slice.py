import numpy as np

def slice_op(data, starts, ends, axes=None, steps=None):
    """
    ONNX Slice オペレータ

    テンソルの一部を切り出す。

    Args:
        data: 入力テンソル
        starts: 各軸の開始インデックスのリスト
        ends: 各軸の終了インデックスのリスト
        axes: 操作する軸のリスト (省略時は全軸)
        steps: 各軸のステップのリスト (省略時は1)

    Returns:
        output: スライスされたテンソル
    """
    ndim = len(data.shape)

    # デフォルト値の設定
    if axes is None:
        axes = list(range(len(starts)))

    if steps is None:
        steps = [1] * len(starts)

    # スライスオブジェクトの作成
    slices = [slice(None)] * ndim

    for i, axis in enumerate(axes):
        slices[axis] = slice(starts[i], ends[i], steps[i])

    return data[tuple(slices)]


if __name__ == "__main__":
    # テスト例
    data = np.arange(24).reshape(4, 6)
    print("元のテンソル:")
    print("形状:", data.shape)
    print(data)

    # 基本的なスライス
    result = slice_op(data, starts=[1, 2], ends=[3, 5], axes=[0, 1])
    print("\nSlice (starts=[1, 2], ends=[3, 5], axes=[0, 1]):")
    print("形状:", result.shape)
    print(result)

    # ステップを使ったスライス
    result = slice_op(data, starts=[0, 0], ends=[4, 6], axes=[0, 1], steps=[2, 2])
    print("\nSlice (steps=[2, 2]):")
    print("形状:", result.shape)
    print(result)
