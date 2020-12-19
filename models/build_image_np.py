# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: build_image_np
@time: 2020/8/4 11:51

Build image numpy array according to vocab and font file

"""

import os
from tqdm import tqdm
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageFont


def dump_image(vocab_file: str, font_file: str, output_file: str, font_size: int = 24):
    """main"""
    font = TTFont(font_file)
    unicode_map = font['cmap'].tables[0].ttFont.getBestCmap()
    font = ImageFont.truetype(font_file, font_size)

    def char_in_ttf(char):
        if len(char) != 1:
            return False
        if ord(char) in unicode_map:
            return True
        else:
            return False

    def get_image(char):
        if char_in_ttf(char):
            char_image = font.getmask(char)
            row, column = char_image.size[::-1]
            # sometimes char_image is larger than font_size
            if row > font_size or column > font_size:
                ratio = min(font_size/row, font_size/column)
                char_image = char_image.resize([int(row * ratio), int(ratio*column)], Image.BILINEAR)
                row, column = char_image.size[::-1]
            char_image = np.asarray(char_image).reshape((row, column))
            # pad equally on each direction
            if row != font_size or column != font_size:
                pad_image = np.zeros([font_size, font_size])
                left_pad = (font_size - column) // 2
                top_pad = (font_size - row) // 2
                pad_image[top_pad: top_pad + row, left_pad: left_pad + column] = char_image
                return pad_image
            return char_image
        return np.zeros([font_size, font_size])

    with open(vocab_file) as fin:
        vocab = fin.readlines()

    vocab_size = len(vocab)
    output_array = np.zeros([vocab_size, font_size, font_size])
    for idx, char in tqdm(enumerate(vocab)):
        char_image = get_image(char.strip())
        output_array[idx] = char_image

    np.save(output_file, output_array)


def main():
    """main"""
    font_dir = "/data/nfsdata2/nlp_application/datasets/fonts/chinese_scripts"
    font_size = 24
    font_files = [
        os.path.join(font_dir, relative_path) for relative_path in [
            'bronzeware_script/HanYiShouJinShuFan-1.ttf',
            'cjk/NotoSansCJKsc-Regular.otf',
            'seal_script/方正小篆体.ttf',
            'tablet_script/WenDingHuangYangJianWeiTi-2.ttf',
            'regular_script/STKAITI.TTF',
            'cursive_script/行草字体.ttf',
            'clerical_script/STLITI.TTF',
            'cjk/STFANGSO.TTF',
            'clerical_script/方正古隶繁体.ttf',
            'regular_script/STXINGKA.TTF'
        ]
    ]
    vocab_file = "/data/nfsdata2/nlp_application/models/bert/shannon_bert/common_crawl_v3/vocab.txt"

    for font_file in font_files:
        output_file = font_file+str(font_size)
        dump_image(vocab_file=vocab_file, font_file=font_file, output_file=output_file, font_size=font_size)


def visualize(
    np_file,
    vocab_file="/data/nfsdata2/nlp_application/models/bert/shannon_bert/common_crawl_v3/vocab.txt"
):
    """load preprocessed np file for visualization"""
    char2idx = {}
    with open(vocab_file) as fin:
        for idx, line in enumerate(fin):
            char = line.strip()
            char2idx[char] = idx
    array = np.load(np_file)
    while True:
        input_char = input("input char:")
        try:
            image_array = array[char2idx[input_char]]
            image = Image.fromarray(image_array)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(f"/data/yuxian/tmp/glyph/{input_char}.png")
        except Exception as e:
            print(e)


if __name__ == '__main__':
    # main()
    visualize("/data/nfsdata2/nlp_application/datasets/fonts/chinese_scripts/clerical_script/方正古隶繁体.ttf24.npy")
