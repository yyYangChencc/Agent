# IAC v2 FourForums 平台保真度审计

## 审计输入

- SQL 压缩包：`D:\agent\Agent\data\iac_v2\fourforums_no_parse_2016_05_18.sql.gz`
- 文件大小：`171653459` 字节
- SHA-256：`e26e6d27a5fc88ee5837b3838f099f8268d42fe50fbbd9cd88b66dfe7744dd9c`
- 完整建表数量：`21`

## 结论

FourForums SQL 可以直接还原作者、讨论、帖子时间线、父帖、引用、话题及人工立场标注。
SQL 不提供原平台关注关系、推荐排序、曝光日志、点赞、点踩或转发事件。
当前场景中的关注关系、线上信任和线下信任均由项目代码生成，必须标记为推导设置。

## 原始数据能力

| 状态 | 能力 | 精确证据 |
|---|---|---|
| `restored` | 发帖者身份 | `author(author_id, username)` |
| `restored` | 讨论页面与发起者 | `discussion(discussion_id, url, title, initiating_author_id)` |
| `restored` | 发帖时间线 | `post(discussion_id, post_id, author_id, creation_date, text_id)` |
| `restored` | 父帖回复关系 | `post(discussion_id, post_id, parent_post_id, parent_missing)` |
| `restored` | 引用及引用来源 | `quote(discussion_id, post_id, quote_index, parent_quote_index, text_id, source_discussion_id, source_post_id, source_start, source_end, truncated, altered)` |
| `restored` | 讨论话题与话题立场定义 | `discussion_topic(discussion_id, topic_id); topic(topic_id, topic); topic_stance(topic_id, topic_stance_id, stance)` |
| `restored` | 人工标注的作者立场 | `mturk_author_stance(discussion_id, author_id, topic_id, topic_stance_id_1, topic_stance_votes_1, topic_stance_id_2, topic_stance_votes_2, topic_stance_votes_other)` |
| `missing` | 原平台关注关系 | `无` |
| `missing` | 原平台推荐算法、特征、分数与排序结果 | `无` |
| `missing` | 原平台曝光、浏览和点击日志 | `无` |
| `missing` | 原平台点赞、点踩和转发事件 | `无` |

## 当前项目保真度

| 状态 | 项目项 | 依据 |
|---|---|---|
| `derived` | `follow_graph` | 场景函数 _follow_edges 生成；FourForums SQL 未提供原平台关注关系；代码行 307 |
| `derived` | `online_trust` | 场景函数 _online_trust 生成；不是原平台字段；代码行 349 |
| `derived` | `offline_trust` | 场景函数 _offline_trust 生成；不是原平台字段；代码行 366 |
| `missing` | `recommendation` | 当前精确入口为 _apply_recommendation_placeholder，配置默认值为 False；代码行 81 |
| `derived` | `exposure_instrumentation` | 项目观测日志不等于原平台曝光数据；代码行 23 |

## 原始 SQL 表

- `author`：`author_id, username`
- `dataset_metadata`：`row_id, metadata_field, metadata_value`
- `discussion`：`discussion_id, url, title, initiating_author_id`
- `discussion_topic`：`discussion_id, topic_id`
- `markup`：`text_id, markup_id, markup_start, markup_end, type, attributes`
- `mturk_2010_dialogue_relation_question`：`question_id, title, task_id, is_scalar, has_unsure, question, low_value_label, high_value_label`
- `mturk_2010_p123_average_response`：`page_id, tab_number, num_annots, disagree_agree, disagree_agree_unsure, attacking_respectful, attacking_respectful_unsure, emotion_fact, emotion_fact_unsure, nasty_nice, nasty_nice_unsure, sarcasm_yes, sarcasm_no, sarcasm_unsure`
- `mturk_2010_p123_entry`：`page_id, tab_number, p123_triple_id, context_triple_index, response_triple_index`
- `mturk_2010_p123_post`：`p123_triple_id, triple_index, discussion_id, post_id, presented_text, presented_text_term_removed, term, topic`
- `mturk_2010_p123_worker_response`：`page_id, tab_number, workerid, response_number, disagree_agree, disagree_agree_unsure, attacking_respectful, attacking_respectful_unsure, emotion_fact, emotion_fact_unsure, nasty_nice, nasty_nice_unsure, sarcasm`
- `mturk_2010_qr_entry`：`page_id, tab_number, discussion_id, post_id, quote_index, response_text_end, presented_quote, presented_response, term, topic`
- `mturk_2010_qr_task1_average_response`：`page_id, tab_number, num_annots, disagree_agree, disagree_agree_unsure, attacking_respectful, attacking_respectful_unsure, emotion_fact, emotion_fact_unsure, nasty_nice, nasty_nice_unsure, sarcasm_yes, sarcasm_no, sarcasm_unsure`
- `mturk_2010_qr_task1_worker_response`：`page_id, tab_number, workerid, response_number, disagree_agree, disagree_agree_unsure, attacking_respectful, attacking_respectful_unsure, emotion_fact, emotion_fact_unsure, nasty_nice, nasty_nice_unsure, sarcasm`
- `mturk_2010_qr_task2_average_response`：`page_id, tab_number, num_disagree, num_annots, agree, defeater_undercutter, defeater_undercutter_unsure, negotiate_attack, negotiate_attack_unsure, personal_audience, personal_audience_unsure, questioning_asserting, questioning_asserting_unsure`
- `mturk_2010_qr_task2_worker_response`：`page_id, tab_number, workerid, response_number, agree, defeater_undercutter, defeater_undercutter_unsure, negotiate_attack, negotiate_attack_unsure, personal_audience, personal_audience_unsure, questioning_asserting, questioning_asserting_unsure`
- `mturk_author_stance`：`discussion_id, author_id, topic_id, topic_stance_id_1, topic_stance_votes_1, topic_stance_id_2, topic_stance_votes_2, topic_stance_votes_other`
- `post`：`discussion_id, post_id, author_id, creation_date, parent_post_id, parent_missing, text_id`
- `quote`：`discussion_id, post_id, quote_index, parent_quote_index, text_offset, text_id, source_discussion_id, source_post_id, source_start, source_end, truncated, altered`
- `text`：`text_id, text`
- `topic`：`topic_id, topic`
- `topic_stance`：`topic_id, topic_stance_id, stance`

## 后续输入要求

若要声称还原原平台关注或推荐设置，必须补充原平台数据库表、平台代码、技术文档或可复核抓包。
在补充证据前，后续实验应分别命名为“项目生成关注图”和“项目实现推荐策略”，不能命名为原平台还原。
