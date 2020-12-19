# encoding: utf-8
"""
@author: Yuxian Meng
@contact: yuxian_meng@shannonai.com

@version: 1.0
@file: ltp_parser
@time: 2020/7/6 21:27

    这一行开始写关于本文件的说明与解释
"""


import os
import re
import jieba
from typing import List, Dict, Any, Tuple

from pyltp import Segmentor, Postagger, Parser
from nlpc.edit_locator import locate_string_lcs_aligns
from utils.singleton import MetaSingleton


class JiebaSegmentor():
    """Jieba Segmentor"""
    def __init__(self, workers=1):
        self.tokenizer = jieba.Tokenizer()
        jieba.initialize()
        jieba.enable_parallel(workers)

    def segment(self, sentence):
        return self.tokenizer.cut(sentence)


class LtpModel(metaclass=MetaSingleton):
    """使用PyLPT做依存分析"""

    def __init__(self, model_path="/data/nfsdata2/nlp_application/models/ltp/ltp_data_v3.4.0", jieba_cut=False, workers=1):
        ltp_data_dir = os.path.join(model_path)
        if jieba_cut:
            print("Using jieba cut")
            self.segmentor = JiebaSegmentor(workers=workers)
        else:
            self.segmentor = Segmentor()
            self.segmentor.load(os.path.join(ltp_data_dir, "cws.model"))
        self.postagger = Postagger()
        self.postagger.load(os.path.join(ltp_data_dir, "pos.model"))
        self.parser = Parser()
        self.parser.load(os.path.join(ltp_data_dir, "parser.model"))
        self.space_re = re.compile("\s")

    def parse(self, raw_sentence: str) -> Dict[str, Any]:
        """parse"""
        words_tags = self.tokenize(raw_sentence)
        words = [word.term for word in words_tags]
        postags = self.postagger.postag(words)
        arcs = self.parser.parse(words, postags)

        sentence_dependency = dict()
        sentence_dependency['words'] = words
        sentence_dependency['arcs'] = [
            (arc.head, arc.relation) for arc in arcs]
        sentence_dependency['postags'] = list(postags)
        sentence_dependency['offsets'] = [w.begin for w in words_tags]
        return sentence_dependency

    def parse_batch(self, **kwargs) -> List[Dict[str, Any]]:
        """parse batch"""
        raw_sentences = kwargs["raw_sentences"]
        depends = []
        for sentence in raw_sentences:
            sentence_dependency = self.parse(sentence)
            depends.append(sentence_dependency)
        return depends

    def tokenize(self, sentence: str) -> List[Tuple[int, int, str]]:
        """模仿CwsPreprocessor的接口获取分词"""
        output = []
        words = list(self.segmentor.segment(sentence))
        # 构造原本char_idx到pytlp_idx的映射(因为它们去除了空格)
        ltp_sentence = "".join(words)
        aligns = locate_string_lcs_aligns(source=sentence, target=ltp_sentence)

        ltp_idx2origin_idx = {ltp_idx: origin_idx for origin_idx, ltp_idx in aligns}

        ltp_offset = 0
        for word in words:
            length = len(word)
            ltp_start = ltp_offset
            ltp_end = ltp_offset + len(word)
            origin_start = ltp_idx2origin_idx[ltp_start]
            origin_end = ltp_idx2origin_idx[ltp_end-1] + 1
            output.append((origin_start, origin_end, sentence[origin_start: origin_end]))
            ltp_offset += length
        return output

    @staticmethod
    def compute_offsets(words: List[str], raw_sentence: str) -> List[int]:
        """
        根据分词结果与原句计算每个词得offset
        Note: pyltp分词时直接忽略了空格，所以需要.index匹配一下
        """
        try:
            offset = 0
            offsets = []
            stack = words.copy()
            stack.reverse()
            while stack:
                word = stack.pop()
                idx = raw_sentence[offset:].index(word)
                offsets.append(idx + offset)
                offset = idx + offset + len(word)
        except Exception as e:
            print(msg=f"pyltp compute offsets bug with inputs: {raw_sentence}")
            offset = 0
            offsets = []
            for word in words:
                offsets.append(offset)
                offset += len(word)
        return offsets

    def offsets2cws_id(self, sentence: str) -> Dict[int, int]:
        """map from origin char offset to cws_offset"""
        offsets2cws = dict()
        ltp_tokens = self.tokenize(sentence)
        for token_idx, (start, end, token) in enumerate(ltp_tokens):
            for idx in range(start, end):
                offsets2cws[idx] = token_idx
        return offsets2cws
