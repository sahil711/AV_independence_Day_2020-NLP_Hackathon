import torch
from torch.utils.data import Dataset


class BertDataV1(Dataset):
    '''
    This version we treat both the title and abstract as two separate sentences
    and use the special tokens to make combine, similar to what is done in bert
    for sentence pair classification task (pre-training)
    '''

    def __init__(self, tokenizer, df, target_cols=None):

        self.df = df
        self.tokenizer = tokenizer
        self.target_cols = target_cols

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        abstract = self.df['ABSTRACT'].iloc[idx]
        title = self.df['TITLE'].iloc[idx]
        output = self.tokenizer(title, abstract, truncation=True,
                                add_special_tokens=True, padding='max_length')

        if self.target_cols is not None:
            for cols in self.target_cols:
                output[cols] = self.df[cols].iloc[idx]

        output = {x: torch.tensor(output[x], dtype=torch.long) for x in output}

        return output


class BertDataV2(Dataset):
    '''
    This version we treat both the title and abstract as a single source
    '''

    def __init__(self, tokenizer, df, target_cols=None):

        self.df = df
        self.tokenizer = tokenizer
        self.target_cols = target_cols

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        text = self.df['text'].iloc[idx]
        output = self.tokenizer(text, truncation=True,
                                add_special_tokens=True, padding='max_length')

        if self.target_cols is not None:
            for cols in self.target_cols:
                output[cols] = self.df[cols].iloc[idx]

        output = {x: torch.tensor(output[x], dtype=torch.long) for x in output}

        return output
