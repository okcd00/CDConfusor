## 相关数据

### vocab
> 在正常中文 BERT 的字典 vocab 基础上，追加一些常见的中文标点。    
> 另外，为错别字任务追加额外的 “命名实体” 和 “冗余字” 的特殊 token。

### vocab_pinyin
> 为目前的全部可用拼音做一个字典
> 为非汉字准备 `[UNK]` token，为长度对齐准备 `[PAD]` token。

### pinyin2tokens
> 借助 `pypinyin` 第三方库收集每种拼音对应哪些汉字（汉字仅考虑 vocab 中的所有汉字） 