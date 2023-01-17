## 相关数据

### vocab
> Based on the origin BERT vocab, add some punctuation marks like Chinese quotes.   
> In addition, we add extra NER marks and Redundant marks for other research。

### vocab_pinyin
> A dict for all pinyin strings.  
> If a token is not a Chinese character, its pinyin will be `[UNK]`，and `[PAD]`'s pinyin is also the `[PAD]` token。

### pinyin2tokens(id)
> With the help of `pypinyin` , collect all characters (in BERT vocab) with the same pinyin string.

### spellGraphs
> take the character-level confusion sets from [ACL2020SpellGCN/SpellGCN](https://github.com/ACL2020SpellGCN/SpellGCN/blob/master/data/gcn_graph.ty_xj/spellGraphs.txt).