from .collator import *
from bbcm.utils.text_utils import (
    is_chinese_char, is_control, is_whitespace, is_punctuation, 
    convert_to_unicode, clean_text
)

class DynamicDataCollatorForCsc(DataCollatorForCsc):
    def __init__(self, tokenizer, augmentation=True, ner_model='stanford', raw_first_epoch=False, random_pos=True, debug=False):
        super(DynamicDataCollatorForCsc, self).__init__(tokenizer)
        self.first_epoch = raw_first_epoch  # the first epoch remains the origin labels. 
        self.random_pos = random_pos  # random position for augmentation
        self.augmentation = augmentation
        self.min_wrong_positions = 1
        self.min_wrong_positions_rate = 0.05
        self.confusor = default_confusor()
        print("Dynamic Data Collator Init\n"
              f"with {100. * self.min_wrong_positions_rate}% faulty positions.")

        self.ner_model = self.init_ner_model(model=ner_model) if ner_model and self.augmentation else None
        self.invalid_tags = ['PERSON', 'NUMBER']  # Location, Person, Organization, Money, Percent, Date, Time
        self.debug = debug

    def init_ner_model(self, model):
        if isinstance(model, str) and model.lower() in ['stanford', 'stanfordnlp']:
            from stanfordcorenlp import StanfordCoreNLP
            stanford_model_path = r'/home/chendian/download/stanford-corenlp-4.2.2/'
            print(f"Now loading NER model from {stanford_model_path}")
            stanford_model = StanfordCoreNLP(stanford_model_path, lang='zh', quiet=True)
            self.ner_model = stanford_model
        return self.ner_model

    def stanford_ner(self, text):
        # from urllib.parse import quote
        # fix the issue with '%' sign for StanfordNLP
        # data=quote(text),  # quote may bring length-changing issues
        r_dict = self.ner_model._request(
            annotators='ner', 
            data=text.replace("%", "％"))
        words = []
        ner_tags = []  # ['O', 'MISC', 'LOCATION', 'GPE', 'FACILITY', 'ORGANIZATION', 'DEMONYM', 'PERSON']
        positions = []
        for s in r_dict['sentences']:
            for token in s['tokens']:
                # "可"
                words.append(token['originalText'])
                # "O"
                ner_tags.append(token['ner'])
                # (2, 3) 
                positions.append((token['characterOffsetBegin'], 
                                  token['characterOffsetEnd']))
        return list(zip(words, ner_tags, positions))

    def generate_word_offsets(self, ct):
        # using pinyin_util's segmentation
        word_offsets = {}
        word_indexes = [0]
        words = self.confusor.pu.segmentation(ct)
        for w in words:
            wc = [_i for _i in range(word_indexes[-1], word_indexes[-1] + len(w))]
            word_offsets.update({_i: wc for _i in wc})
            word_indexes.append(len(w) + word_indexes[-1])
        return word_offsets

    def get_valid_position_mask(self, text):        
        # using stanford ner's segmentation
        # a list of [original_text, ner_tag, (begin, end)]
        results = self.stanford_ner(text)
        
        # the mask is char-level only for augmentation, not the same with det_label's mask
        text_len = len(text)
    
        # init masks
        word_offsets = {}
        chn_mask = np.ones(text_len, dtype=int)
        ner_mask = np.zeros(text_len, dtype=int)

        # generate chn_mask
        for w_idx, w in enumerate(convert_to_unicode(text)):
            if is_whitespace(w) or is_control(w):
                chn_mask[w_idx] = 0
            # TODO: commas are easy to correct, maybe we can also correct them
            elif (w in self.confusor.PUNC_LIST) or is_punctuation(w):
                chn_mask[w_idx] = 0
            elif not is_chinese_char(ord(w)):
                chn_mask[w_idx] = 0
        
        # generate word_offsets and ner_mask
        valid_flag = True
        if results[-1][-1][-1] > text_len:  # the last word's position's ending_pos
            print("Detected an invalid text:", text)
            valid_flag = False
        for w_idx, (w, tag, position) in enumerate(results):
            tokens_in_word = list(range(position[0], position[1]))  # (l, r) from stanfordNLP
            word_offsets.update(
                {token: tokens_in_word for token in tokens_in_word})
            # cur_chn_mask = [1 if is_chinese_char(ord(token)) else 0 for token in convert_to_unicode(w)]
            # if (position[1] - position[0]) == len(w) == len(cur_chn_mask):
            #     chn_mask[position[0]: position[1]] = cur_chn_mask
            # else:
            #     print(w, tag, position, 
            #           (position[1] - position[0], len(w), len(cur_chn_mask)))
            #     print(f"Unaligned length between word <{w}> and chn_mask <{cur_chn_mask}>")
            if valid_flag and (tag in self.invalid_tags):
                ner_mask[position[0]: position[1]] = 1

        return word_offsets, ner_mask.tolist(), chn_mask.tolist()

    def change_words(self, word, correct_word=None, change_part=False, sentence=None):
        # maybe the word-changing method will relate to the sentence context.
        if len(word) == 1:
            try:
                candidates = self.confusor(word)
            except Exception as e:
                print("Error word text:", word)
                print(str(e))
                return word  # leave it un-changed
            # candidates += self.confusor.char_confusion_set.get(word, [])
            # can = deepcopy(candidates)
        elif len(word) > 4 or change_part:  
            # we only have embeddings for words shorter than 4 tokens
            change_part = True
            change_position = random.choice(range(len(word)))
            changed_part = self.change_words(
                word=word[change_position: change_position+1], 
                correct_word=correct_word[change_position: change_position+1] if correct_word else None, 
                sentence=sentence)
            # print(f"{word[change_position: change_position+1]} -> {changed_part}")
            return f"{word[:change_position]}{changed_part}{word[change_position+1:]}"
        else:
            try:
                candidates = self.confusor(word)
            except Exception as e:
                print("Error word text:", word)
                print(str(e))
                return word  # leave it un-changed
            # for can_word in candidates:
            #     if can_word in white_list:
            #         candidates.remove(can_word)
        if correct_word and correct_word in candidates:
            candidates.remove(correct_word)
        ret_word = random.choice(candidates) if candidates else word
        return ret_word

    def random_wrong_ids(self, ct, wrong_id, word_level=True, word_offsets=None, valid_position_mask=True):
        ot = deepcopy(ct)
        text_len = ot.__len__()

        ner_mask, chn_mask = [], []
        if valid_position_mask:
            word_offsets, ner_mask, chn_mask = self.get_valid_position_mask(ct)

        def valid_position(idx):
            if chn_mask: 
                if chn_mask[idx] == 0:
                    return False  # blank token also ends here
            if ner_mask:
                try:
                    if ner_mask[idx] == 1:  
                        # [1 1 0 0 0 1], 1 means a token in NER
                        return False
                except Exception as e:
                    print(e)
                    print(ct, len(ct))
                    print(ner_mask, idx)
                    return False 
            elif not is_chinese_char(ord(ct[idx])):
                return False
            return True

        # only chinese char can be augmented
        candidate_position_mask = [1 if valid_position(i) else 0 for i in range(text_len)]
        candidate_position = [i for i in range(text_len) if candidate_position_mask[i]]
        char_count = len(candidate_position)
        w_idx, word_count, lst = 0, 0, []
        word_record = {}
        for w_idx, lst in enumerate(word_offsets.values()):
            if tuple(lst) in word_record:
                continue
            word_record[tuple(lst)] = 1
            for _p in lst:
                if _p >= len(candidate_position_mask):
                    print("Failed for counting words")
                    print(set([tuple(lst) for lst in word_offsets.values()]))
                    print(candidate_position_mask, lst, w_idx)
                    break
                if candidate_position_mask[_p] == 1:
                    word_count += 1
                    break
                # word_count = len(set([tuple(lst) for lst in word_offsets.values() 
                #                       if sum([candidate_position_mask[_p] for _p in lst]) > 0]))
        if word_level:
            changed_word_counts = math.ceil(
                self.min_wrong_positions_rate * word_count)
        else:  # char_level
            changed_word_counts = math.ceil(
                self.min_wrong_positions_rate * char_count)
        wrong_position_counts = max(len(wrong_id),
                                    self.min_wrong_positions,
                                    changed_word_counts)
        if self.debug:
            print("wrong_position_counts:", wrong_position_counts)

        # In case of the number candidate position is too small
        if len(candidate_position) == 0:
            wrong_position_counts = 0    
        elif wrong_position_counts / len(candidate_position) > 0.4:
            wrong_position_counts = math.floor(len(candidate_position) * 0.4)
        try:
            wrong_ids = sorted(random.sample(candidate_position,
                                             wrong_position_counts))
        except Exception as e:
            print(f"Sample Error for {str(e)}")
            print(candidate_position, wrong_position_counts)
            wrong_ids = []

        return sorted(wrong_ids), word_offsets, chn_mask, ner_mask

    def sample_augment_single(self, ot, ct, wrong_id, valid_position_mask=True, 
                              random_pos=None, word_level=True, detail=False):
        # wr_ids here should be char-level
        o_text, c_text, wr_ids = deepcopy(ot), ct, deepcopy(wrong_id)
        word_offsets = None
        if random_pos is not None:
            self.random_pos = random_pos
        if self.random_pos:  # change another position to augment
            wr_ids, word_offsets, chn_mask, ner_mask = self.random_wrong_ids(
                ct=c_text, wrong_id=wr_ids, word_level=word_level,
                valid_position_mask=valid_position_mask)
            if self.debug:
                print(wr_ids, word_offsets)
                print(chn_mask, ner_mask)
        if word_offsets is None:
            word_offsets = self.generate_word_offsets(c_text)

        done_wid_list = []  # done wid in the same word.
        for wid in wr_ids:
            if wid in done_wid_list:
                continue
            word_ids = [wid]
            if word_level:  # change the word or the character only
                # half for the char only, half for the word it located in
                if random.random() > 0.5:  
                    # change the word it relates
                    related_word = sorted(word_offsets.get(wid, [wid]))  # token aug -> word aug
                    # a non-Chinese in this word, such as "2012年"
                    if 0 not in [chn_mask[_wd] for _wd in related_word]:
                        word_ids = related_word
                        done_wid_list.extend(word_ids)
            _word = ''.join([o_text[_i] for _i in word_ids])
            _correct_word = ''.join([c_text[_i] for _i in word_ids])
            if self.debug:
                print("select the word:", _word, word_ids)
            _changed_word = self.change_words(
                word=_word, correct_word=_correct_word, sentence=c_text)
            # change ori_text here
            _l, _r = word_ids[0], word_ids[-1] + 1
            o_text = f"{o_text[:_l]}{_changed_word}{o_text[_r:]}"
            # if _changed_word != c_text[wid]: current_wids.append(wid)
        current_wids = [_id for _id in
                        range(len(c_text)) if c_text[_id] != o_text[_id]]
        if detail:
            detail_dict = {
                'word_offsets': word_offsets, 
                'chn_mask': chn_mask, 
                'ner_mask': ner_mask
            }
            return o_text, c_text, current_wids, detail_dict
        return o_text, c_text, current_wids

    def sample_augment(self, ori_text, cor_text, wrong_ids, random_pos=None, word_level=True):
        ori_text_case, cor_text_case, wrong_ids_case = [], cor_text, []
        # ori_text, cor_text, wrong_ids are all lists
        if random_pos is not None:
            self.random_pos = random_pos
        for o, c_text, _ in zip(ori_text, cor_text, wrong_ids):
            o_text, _, current_wids = self.sample_augment_single(
                ot=o, ct=c_text, wrong_id=[],
                random_pos=self.random_pos, word_level=word_level)
            ori_text_case.append(o_text)
            wrong_ids_case.append(current_wids)
        return ori_text_case, cor_text_case, wrong_ids_case

    def samples(self):
        return [sample for s_idx, sample in enumerate(self)]

    def load_csc_dataset(self, csc_data_path, csc_data_type='json'):
        csc_origin_data = []
        if os.path.exists(csc_data_path):
            # load from pure text lines
            dataset_type = PureTextDataset
            if 'json' in csc_data_type:
                # csc_origin_data = json.load(open(csc_data_path, 'r'))
                dataset_type = CscDataset
            csc_origin_data = dataset_type(csc_data_path)
        return csc_origin_data

    def generate_csc_augmented_samples(self, csc_origin_data, csc_data_path=None,
                                       csc_data_type='json', random_pos=None, detail=False):
        """

        :param csc_origin_data: a dataset from {PureTextDataset, CscDataset}
        :param csc_data_path:
        :param random_pos:
        :return:
        """
        if csc_data_path and (csc_origin_data is None):
            csc_origin_data = self.load_csc_dataset(
                csc_data_path=csc_data_path, csc_data_type=csc_data_type)
        if random_pos is not None:
            self.random_pos = random_pos
        augmented_samples = []
        for s_idx, sample in enumerate(csc_origin_data):
            # o = sample['original_text']
            # c = sample['correct_text']
            # w = sample['wrong_ids']
            o, c, w = sample
            sample_dict = {}
            if detail:
                o_text, c_text, current_wids, detail_dict = self.sample_augment_single(
                    ot=o, ct=c, wrong_id=w, random_pos=self.random_pos, detail=True)
                sample_dict.update(detail_dict)
            else:
                o_text, c_text, current_wids = self.sample_augment_single(
                    ot=o, ct=c, wrong_id=w, random_pos=self.random_pos, detail=False)
            sample_dict.update({
                'id': f"{s_idx}",  # 'id': sample.get('id'),
                'original_text': o_text,
                'wrong_ids': current_wids,
                'correct_text': c_text,
            })
            augmented_samples.append(sample_dict)
        return augmented_samples

    def generate_csc_augmented_samples_from_text_dir(self, csc_data_path, output_dir=None, overwrite=False):
        """
        generate from a too-large pure-text corpus.
        :param text_path_pattern:
        :return:
        """
        if output_dir is None:
            output_dir = '/data/chendian/findoc_csc_samples_test_210917'
        csc_origin_data = self.load_csc_dataset(
            csc_data_path=csc_data_path, csc_data_type='text')
        augmented_samples = []
        # n_samples = csc_origin_data.__len__()
        last_file = ""
        for s_idx, sample in tqdm(enumerate(csc_origin_data)):
            # o = sample['original_text']
            # c = sample['correct_text']
            # w = sample['wrong_ids']
            o, c, w = sample
            # all mask should be real-time generated 
            o_text, c_text, current_wids = self.sample_augment_single(
                ot=o, ct=c, wrong_id=w, random_pos=True)

            cur_file = f"{csc_origin_data.current_file_index}"
            if cur_file != last_file:
                # print(last_file, cur_file)
                if last_file != "":
                    out_file = f"{output_dir}/findoc_{last_file}.json"
                    # if os.path.exists(out_file):  continue  # no overwrite
                    dump_json(augmented_samples, out_file)
                    augmented_samples = []
                last_file = '{}'.format(cur_file)
            augmented_samples.append({
                'id': f"{cur_file}_{s_idx}",
                'original_text': o_text,
                'wrong_ids': current_wids,
                'correct_text': c_text,
            })
        else:
            out_file = f"{output_dir}/findoc_{last_file}.json"
            # if os.path.exists(out_file):  continue  # no overwrite
            dump_json(augmented_samples, out_file)

    def __call__(self, data):
        # return the original samples for the first epoch (optional)
        if self.augmentation and not self.first_epoch:
            # char-level augmentation
            ori_texts, cor_texts, _ = self.sample_augment(*zip(*data))
        else:
            ori_texts, cor_texts, _ = zip(*data)
        
        self.first_epoch = False
        ori_texts = [clean_text(t) for t in ori_texts]
        cor_texts = [clean_text(t) for t in cor_texts]

        # token-level modeling
        encoded_texts = [self.tokenizer.tokenize(t) for t in ori_texts]
        encoded_cor_texts = [self.tokenizer.tokenize(t) for t in cor_texts]

        max_len = max([len(t) for t in encoded_texts]) + 2  # CLS & SEP
        det_labels = torch.zeros(len(ori_texts), max_len).long()
        for i, (encoded_text, encoded_cor_text) in enumerate(zip(encoded_texts, encoded_cor_texts)):
            # auto-matically generate wrong_ids
            wrong_ids = [_i for _i, (_ot, _ct) in enumerate(zip(encoded_text, encoded_cor_text)) 
                         if _ot != _ct]
            # pprint([(_i, _ot, _ct) for _i, (_ot, _ct) in enumerate(zip(encoded_text, encoded_cor_text))])
            # print(wrong_ids)
            for idx in wrong_ids:
                margins = []
                for word in encoded_text[:idx]:
                    if word == '[UNK]':
                        break
                    if word.startswith('##'):
                        margins.append(len(word) - 3)
                    else:
                        margins.append(len(word) - 1)
                margin = sum(margins)
                move = 0
                while True:
                    try:
                        if abs(move) < margin:
                            move -= 1
                        elif idx + move >= len(encoded_text):
                            move -= 1
                        elif encoded_text[idx + move].startswith('##'):
                            move -= 1
                        else:
                            break
                    except Exception as e:
                        print("Failed for generating det_labels.", str(e))
                        print(len(encoded_text), encoded_text)
                        print(idx, move, margin)
                        break
                det_labels[i, idx + move + 1] = 1
        return ori_texts, cor_texts, det_labels


if __name__ == "__main__":
    from tqdm import tqdm
    from bbcm.config import cfg
    from bbcm.utils import get_abs_path
    config_file='csc/train_SoftMaskedBert.yml'
    cfg.merge_from_file(get_abs_path('configs', config_file))

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)

    from bbcm.data.loaders.collator import *
    ddc = DynamicDataCollatorForCsc(tokenizer=tokenizer, augmentation=True, debug=True)
    # ddc = DataCollatorForCsc(tokenizer=tokenizer)
    # print(ddc.change_words('路遥知马力'))
    data = ([('Overall路遥知马力，日久见人心012345！', 'Overall路遥知马力，日久现人心012345！', [9])])
    # item = ddc.sample_augment_single(data)
    # print(item)
    for each in ddc(data):
        print(each)
