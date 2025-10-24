import torch
from torchtext.datasets import Multi30k # *** 更改数据集 ***
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 定义特殊符号
UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN = '<unk>', '<pad>', '<bos>', '<eos>'
special_symbols = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

# 初始化分词器
token_transform = {
    'de': get_tokenizer('spacy', language='de_core_news_sm'),
    'en': get_tokenizer('spacy', language='en_core_web_sm')
}
vocab_transform = {}

def yield_tokens(data_iter, lang, lang_idx):
    """ 
    辅助函数：从数据迭代器中生成token
    data_iter 产生 (de_sent, en_sent) 元组
    lang_idx: 0 表示德语 (de), 1 表示英语 (en)
    """
    for data_sample in data_iter:
        yield token_transform[lang](data_sample[lang_idx])

def load_data_and_vocab():
    """
    加载 Multi30k 数据集并构建词表
    """
    print("加载Multi30k数据集")
    # 加载数据集 (train, valid)
    # Multi30k 迭代器产生 (de_sentence, en_sentence) 格式的元组
    train_iter = Multi30k(split='train', language_pair=('de', 'en'))
    valid_iter = Multi30k(split='valid', language_pair=('de', 'en'))
    
    # 构建词表
    print("构建词表中...")
    
    # 构建德语 (de) 词表
    train_iter_de = Multi30k(split='train', language_pair=('de', 'en'))
    vocab_transform['de'] = build_vocab_from_iterator(
        yield_tokens(train_iter_de, 'de', 0), # lang_idx=0
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )
    
    # 构建英语词表
    train_iter_en = Multi30k(split='train', language_pair=('de', 'en'))
    vocab_transform['en'] = build_vocab_from_iterator(
        yield_tokens(train_iter_en, 'en', 1), # lang_idx=1
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )
    
    #设置默认索引 (UNK)
    for ln in ['de', 'en']:
        vocab_transform[ln].set_default_index(vocab_transform[ln][UNK_TOKEN])
        
    print(f"DE 词表大小: {len(vocab_transform['de'])}")
    print(f"EN 词表大小: {len(vocab_transform['en'])}")

    return train_iter, valid_iter, vocab_transform

# 批处理函数
def collate_fn(batch):
    """
    处理数据批次：分词、添加BOS/EOS、数值化、填充
    """
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch: # batch 是 (de, en) 元组的列表
        # 源序列 (de)
        src_batch.append(torch.tensor(
            [vocab_transform['de'][BOS_TOKEN]] + 
            [vocab_transform['de'][token] for token in token_transform['de'](src_sample)] +
            [vocab_transform['de'][EOS_TOKEN]]
        ))
        # 目标序列 (en)
        tgt_batch.append(torch.tensor(
            [vocab_transform['en'][BOS_TOKEN]] +
            [vocab_transform['en'][token] for token in token_transform['en'](tgt_sample)] +
            [vocab_transform['en'][EOS_TOKEN]]
        ))

    # 填充
    src_batch = pad_sequence(src_batch, padding_value=vocab_transform['de'][PAD_TOKEN], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=vocab_transform['en'][PAD_TOKEN], batch_first=True)
    
    return src_batch, tgt_batch

def get_dataloaders(batch_size=32):
    """
    获取训练和验证的DataLoader
    """
    train_iter, valid_iter, vocab_transform = load_data_and_vocab()
    
    PAD_IDX_SRC = vocab_transform['de'][PAD_TOKEN]
    PAD_IDX_TGT = vocab_transform['en'][PAD_TOKEN]

    train_dataloader = DataLoader(list(train_iter), batch_size=batch_size, collate_fn=collate_fn)
    valid_dataloader = DataLoader(list(valid_iter), batch_size=batch_size, collate_fn=collate_fn)
    
    return train_dataloader, valid_dataloader, vocab_transform, PAD_IDX_SRC, PAD_IDX_TGT