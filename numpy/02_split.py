import numpy as np

def split(input_tensor, split_sizes=None, axis=0, num_outputs=None):
    """
    ONNX Split オペレータ

    テンソルを指定された軸に沿って分割する。

    Args:
        input_tensor: 入力テンソル
        split_sizes: 各分割のサイズのリスト (省略時は均等分割)
        axis: 分割する軸 (デフォルト: 0)
        num_outputs: 出力数 (split_sizes省略時に使用)

    Returns:
        outputs: 分割されたテンソルのリスト
    """
    if split_sizes is not None:
        # 指定されたサイズで分割
        indices = np.cumsum(split_sizes)[:-1]
        return np.split(input_tensor, indices, axis=axis)
    elif num_outputs is not None:
        # 均等分割
        return np.split(input_tensor, num_outputs, axis=axis)
    else:
        # デフォルト: 2つに均等分割
        return np.split(input_tensor, 2, axis=axis)


if __name__ == "__main__":
    # テスト例
    X = np.arange(12).reshape(4, 3)
    print("元のテンソル:")
    print("形状:", X.shape)
    print(X)

    # 均等に2分割
    outputs = split(X, num_outputs=2, axis=0)
    print("\nSplit (num_outputs=2, axis=0):")
    for i, out in enumerate(outputs):
        print(f"出力{i}: 形状 {out.shape}")
        print(out)

    # 不均等分割
    outputs = split(X, split_sizes=[1, 3], axis=0)
    print("\nSplit (split_sizes=[1, 3], axis=0):")
    for i, out in enumerate(outputs):
        print(f"出力{i}: 形状 {out.shape}")
        print(out)
