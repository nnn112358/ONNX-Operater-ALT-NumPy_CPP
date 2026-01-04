import numpy as np
from scipy import ndimage

def resize(X, scales=None, sizes=None, mode='nearest'):
    """
    ONNX Resize オペレータ

    テンソルをリサイズする。

    Args:
        X: 入力テンソル
        scales: 各次元のスケール係数 (例: [1.0, 1.0, 2.0, 2.0])
        sizes: 出力サイズ (例: [1, 3, 128, 128])
        mode: 補間モード ('nearest', 'linear') デフォルト: 'nearest'

    Returns:
        Y: リサイズされたテンソル
    """
    if scales is not None:
        # スケールを使用してリサイズ
        if mode == 'nearest':
            order = 0
        elif mode == 'linear':
            order = 1
        else:
            order = 0

        return ndimage.zoom(X, scales, order=order)

    elif sizes is not None:
        # サイズを使用してリサイズ
        scales = [sizes[i] / X.shape[i] for i in range(len(sizes))]
        if mode == 'nearest':
            order = 0
        elif mode == 'linear':
            order = 1
        else:
            order = 0

        return ndimage.zoom(X, scales, order=order)

    return X


if __name__ == "__main__":
    # テスト例
    X = np.array([[1, 2], [3, 4]])
    print("元の形状:", X.shape)
    print(X)

    # スケールでリサイズ
    resized = resize(X, scales=[2.0, 2.0], mode='nearest')
    print("\nResize (scales=[2.0, 2.0]):")
    print("形状:", resized.shape)
    print(resized)

    # サイズでリサイズ
    resized = resize(X, sizes=[4, 4], mode='nearest')
    print("\nResize (sizes=[4, 4]):")
    print("形状:", resized.shape)
    print(resized)
