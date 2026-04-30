import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import datasets
import unicodedata
from utils import get_model_identifiers_from_yaml, add_dataset_index


def normalize_eval_text(text, language, unicode_normalization=None, normalize_languages=None):
    if unicode_normalization is None or str(unicode_normalization).lower() in {"", "none", "false"}:
        return text

    if normalize_languages is not None:
        if isinstance(normalize_languages, str):
            normalize_languages = {normalize_languages}
        else:
            normalize_languages = set(normalize_languages)
        if language not in normalize_languages:
            return text

    return unicodedata.normalize(str(unicode_normalization), text)


def convert_raw_data_to_model_format(tokenizer, max_length, question, answer, model_configs, language='en'):
    def _build_tensors(encoded_input_ids, encoded_attention_mask, num_prompt_tokens):
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        pad_length = max_length - len(encoded_input_ids)
        pad_input_ids = encoded_input_ids + [eos_token_id] * pad_length
        pad_attention_mask = encoded_attention_mask + [0] * pad_length

        if len(encoded_input_ids) == max_length:
            label = encoded_input_ids.copy()
        else:
            label = encoded_input_ids + [eos_token_id] + [-100] * (pad_length - 1)

        # Only compute loss on assistant answer tokens.
        for i in range(min(num_prompt_tokens, len(label))):
            label[i] = -100

        return torch.tensor(pad_input_ids), torch.tensor(label), torch.tensor(pad_attention_mask)

    use_chat_template = str(model_configs.get('use_chat_template', "false")).lower() == "true"
    if use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
            raise ValueError("use_chat_template=true but tokenizer has no chat template.")

        user_messages = [{"role": "user", "content": question}]
        full_messages = user_messages + [{"role": "assistant", "content": answer}]

        prompt_text = tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_encoded = tokenizer(
            prompt_text,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
        full_encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )

        return _build_tensors(
            full_encoded["input_ids"],
            full_encoded["attention_mask"],
            len(prompt_encoded["input_ids"]),
        )

    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'][language], \
    model_configs['question_end_tag'], model_configs['answer_tag'][language]
    new_question = question_start_token + question + question_end_token
    new_answer = answer_token + answer
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )

    return _build_tensors(
        encoded["input_ids"],
        encoded["attention_mask"],
        num_question_tokens,
    )


class TextDatasetQAEval(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split=None, question_key='question',
                 answer_key='answer', language='en', unicode_normalization=None, normalize_languages=None):
        super(TextDatasetQAEval, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language = language
        self.unicode_normalization = unicode_normalization
        self.normalize_languages = normalize_languages

        # self.data = datasets.load_dataset(data_path, split)["train"]
        self.data = datasets.load_from_disk(data_path)['train']

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = normalize_eval_text(
            self.data[idx][self.qk],
            self.language,
            self.unicode_normalization,
            self.normalize_languages,
        )
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            answer = normalize_eval_text(answer, self.language, self.unicode_normalization, self.normalize_languages)
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer,
                                                              self.model_configs, self.language)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(), \
            torch.stack(label_list).squeeze(), \
            torch.stack(pad_attention_mask_list).squeeze(), \
            torch.tensor(indices)


class TextDatasetQAStat(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split=None, question_key='question',
                 answer_key='answer', language='en', unicode_normalization=None, normalize_languages=None):
        super(TextDatasetQAStat, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.language=language
        self.unicode_normalization = unicode_normalization
        self.normalize_languages = normalize_languages
        if language == 'en':
            self.data = datasets.load_dataset(data_path, split)["train"]
        else:
            self.data = datasets.load_from_disk(data_path)['train']

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = normalize_eval_text(
            self.data[idx][self.qk],
            self.language,
            self.unicode_normalization,
            self.normalize_languages,
        )
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        language = self.language
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            answer = normalize_eval_text(answer, language, self.unicode_normalization, self.normalize_languages)
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer,
                                                              self.model_configs, language)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(), \
            torch.stack(label_list).squeeze(), \
            torch.stack(pad_attention_mask_list).squeeze(), \
            torch.tensor(indices)


class TextDatasetQA(Dataset):
    def __init__(self, data_path, tokenizer, model_family, max_length=512, split=None, question_key='question',
                 answer_key='answer'):
        super(TextDatasetQA, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # self.data = datasets.load_dataset(data_path, split)["train"]
        self.data = datasets.load_from_disk(data_path)['train']

        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']
        language = self.data[idx]['language']
        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer,
                                                              self.model_configs, language)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(), \
            torch.stack(label_list).squeeze(), \
            torch.stack(pad_attention_mask_list).squeeze(), \
            torch.tensor(indices)


def collate_fn(batch):
    input_ids, attention_masks = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return input_ids, attention_masks


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def custom_data_collator_with_indices(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    indices = [s[3] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask), torch.stack(indices)


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss
