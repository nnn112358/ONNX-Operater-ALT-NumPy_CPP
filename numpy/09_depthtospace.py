import numpy as np

def depthtospace(input_tensor, blocksize):
    """
    ONNX DepthToSpace オペレータ

    チャネル次元を空間次元（H, W）に再配置する。
    (N, C, H, W) -> (N, C / blocksize^2, H * blocksize, W * blocksize)

    Args:
        input_tensor: 入力テンソル (N, C, H, W)
        blocksize: ブロックサイズ

    Returns:
        output: 再配置されたテンソル
    """
    N, C, H, W = input_tensor.shape

    # 新しい形状
    new_C = C // (blocksize ** 2)
    new_H = H * blocksize
    new_W = W * blocksize

    # リシェイプと転置を使って実装
    # (N, C, H, W) -> (N, C//bs^2, bs, bs, H, W)
    temp = input_tensor.reshape(N, new_C, blocksize, blocksize, H, W)

    # (N, C//bs^2, bs, bs, H, W) -> (N, C//bs^2, H, bs, W, bs)
    temp = temp.transpose(0, 1, 4, 2, 5, 3)

    # (N, C//bs^2, H, bs, W, bs) -> (N, C//bs^2, H*bs, W*bs)
    output = temp.reshape(N, new_C, new_H, new_W)

    return output


if __name__ == "__main__":
    # テスト例
    N, C, H, W = 1, 8, 2, 2
    blocksize = 2

    input_tensor = np.arange(N * C * H * W).reshape(N, C, H, W)
    print("入力形状:", input_tensor.shape)
    print("入力チャネル数:", C)

    output = depthtospace(input_tensor, blocksize)
    print("\nDepthToSpace (blocksize=2):")
    print("出力形状:", output.shape)
    print("出力 (チャネル0):")
    print(output[0, 0])
    print("出力 (チャネル1):")
    print(output[0, 1])

    # SpaceToDepth -> DepthToSpace の往復確認
    print("\n\n往復テスト:")
    original = np.arange(1 * 2 * 4 * 4).reshape(1, 2, 4, 4)
    print("元の形状:", original.shape)

    # SpaceToDepthを適用（前のコードから借用）
    from spacetodepth import spacetodepth
    s2d = spacetodepth(original, 2)
    print("SpaceToDepth後の形状:", s2d.shape)

    # DepthToSpaceで戻す
    d2s = depthtospace(s2d, 2)
    print("DepthToSpace後の形状:", d2s.shape)
    print("元のデータと一致:", np.allclose(original, d2s))
