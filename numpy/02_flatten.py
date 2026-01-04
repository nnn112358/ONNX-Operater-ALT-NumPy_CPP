import numpy as np

def flatten(input_tensor, axis=1):
    """
    ONNX Flatten オペレータ

    テンソルを2次元に平坦化する。
    axis より前の次元を第1次元に、axis 以降の次元を第2次元にまとめる。

    Args:
        input_tensor: 入力テンソル
        axis: 平坦化の基準となる軸 (デフォルト: 1)

    Returns:
        output: 平坦化されたテンソル
    """
    shape = input_tensor.shape

    # axis より前の次元のサイズを計算
    dim1 = int(np.prod(shape[:axis]))
    # axis 以降の次元のサイズを計算
    dim2 = int(np.prod(shape[axis:]))

    return np.reshape(input_tensor, (dim1, dim2))


if __name__ == "__main__":
    # テスト例
    X = np.arange(24).reshape(2, 3, 4)
    print("元の形状:", X.shape)
    print(X)

    # axis=1 でフラット化
    flat = flatten(X, axis=1)
    print("\nFlatten (axis=1):")
    print("形状:", flat.shape)
    print(flat)

    # axis=2 でフラット化
    flat = flatten(X, axis=2)
    print("\nFlatten (axis=2):")
    print("形状:", flat.shape)
    print(flat)

    # axis=0 でフラット化
    flat = flatten(X, axis=0)
    print("\nFlatten (axis=0):")
    print("形状:", flat.shape)
