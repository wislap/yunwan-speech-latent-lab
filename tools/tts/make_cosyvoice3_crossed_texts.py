#!/usr/bin/env python3
"""Create a crossed multi-speaker Chinese manifest for CosyVoice3 distillation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
CROSS_LINGUAL_PROMPT_TEXT = (
    "You are a helpful assistant.<|endofprompt|>"
    "在那之后完全收购那家公司，因此保持管理层的一致性，利益与即将加入家族的资产保持一致，"
    "这就是我们有时不买下全部的原因。"
)


TEXTS = {
    "daily": [
        "今天的天气很适合散步，我们可以沿着河边慢慢走一会儿。",
        "请把会议记录整理成三段摘要，并标出需要跟进的事项。",
        "这个方案先不要急着上线，等验证结果稳定以后再决定。",
        "如果明天上午有空，我们可以把剩下的样本一起检查完。",
        "他把钥匙放在桌子左边，然后转身去厨房倒了一杯水。",
        "这段音频听起来很干净，背景里几乎没有明显噪声。",
        "请你稍微放慢一点速度，把最后两个数字再重复一遍。",
        "我们先保存当前版本，然后再尝试新的参数组合。",
    ],
    "wiki": [
        "长江是亚洲最长的河流，流经多个省份，最终注入东海。",
        "太阳系由太阳、行星、卫星、小行星和彗星等天体组成。",
        "印刷术的发展改变了知识传播的方式，也推动了教育普及。",
        "杭州位于中国东南沿海地区，是一座历史悠久的城市。",
        "机器学习模型通常需要大量数据，才能获得较好的泛化能力。",
        "古代丝绸之路连接了东西方贸易，也促进了文化交流。",
        "海洋覆盖了地球表面的大部分区域，对气候系统有重要影响。",
        "显微镜让人类能够观察细胞结构，从而推动了现代生物学。",
    ],
    "numbers": [
        "订单编号是二零二六零五零四，请确认金额为三百七十二元。",
        "火车将在上午九点四十五分出发，预计下午一点二十分到达。",
        "这个文件一共有十六个版本，最新版本保存在第三个文件夹里。",
        "请记录三个数值，分别是零点八五，一点二零，以及三点七六。",
        "他的电话号码后四位是七三九二，地址在北京路二十八号。",
        "今天是二零二六年五月四日，星期一，天气多云转晴。",
        "模型训练了十二万个步骤，最终验证损失下降到零点零六。",
        "这批数据包含一百九十条样本，总时长约为二十七分钟。",
    ],
    "questions": [
        "你觉得这个声线适合做长期的语音助手吗？",
        "如果验证集表现没有提升，我们应该优先检查哪一个模块？",
        "这句话的停顿是不是太短了，听起来会不会有点赶？",
        "我们是否需要加入更多说话人，还是先扩大文本覆盖范围？",
        "你能不能把这段内容改写得更自然一点？",
        "这个发音听起来清楚吗，声调有没有明显错误？",
        "如果只改变语速，模型会不会把它误认为新的声线？",
        "这组样本能不能作为第一版蒸馏数据的烟测集合？",
    ],
    "hard": [
        "北京、上海、广州和深圳，是中国重要的一线城市。",
        "人工智能、语音合成和自动语音识别，经常出现在同一个系统里。",
        "他说：“先别急，我们把证据看完再下结论。”",
        "这个缩写有两种读法，具体应该根据上下文来判断。",
        "在百分之九十五的情况下，简单规则已经足够稳定。",
        "请注意，括号里的内容通常不需要读得太重。",
        "中英文混合时，模型需要保持节奏，不要突然加速。",
        "版本号从一点零升级到一点一，改动很小，但影响很关键。",
    ],
}


LONG10_TEXTS = {
    "daily": [
        "今天下午我们先把会议记录整理完，再检查每一条待办事项的负责人和截止时间，避免明天讨论时遗漏关键问题。",
        "如果这批样本的发音、停顿和音量都比较稳定，我们就可以把它作为第一版蒸馏数据的基准集合来使用。",
        "他把手机放在桌子右上角，认真核对快递单号和收货地址，然后才确认这笔订单没有填写错误。",
        "为了让后续评估更容易复现，我们需要保留文本编号、声线编号、语速条件和生成模型版本等所有元信息。",
        "早高峰结束以后，路上的车辆明显少了很多，司机把车窗打开一点，让清凉的空气慢慢流进车里。",
        "这段录音的背景比较安静，但说话人有几处轻微停顿，所以我们需要先判断这些停顿是否符合自然表达。",
        "晚饭之前她又检查了一遍行李，确认身份证、充电器和两份纸质材料都已经放进了背包里面。",
        "如果实验结果和昨天相比没有明显改善，我们应该先固定数据版本，再逐项排查模型结构和训练参数。",
    ],
    "wiki": [
        "长江流域覆盖范围很广，不同地区的地形、气候和城市分布差异明显，因此相关生态问题也需要分区讨论。",
        "太阳系中的行星围绕太阳运行，而卫星、小行星和彗星等天体共同构成了更加复杂的空间环境。",
        "印刷术的传播降低了书籍复制成本，让知识能够更快进入学校、商铺和普通家庭，进而改变了社会结构。",
        "杭州自古以来就是重要的交通和商业城市，河道、湖泊与街巷共同塑造了它独特的城市风貌。",
        "机器学习模型通常依赖大量样本进行训练，但数据质量、标签一致性和评估方式同样会影响最终表现。",
        "显微镜的出现让研究者能够观察细胞和微小结构，从而推动医学、生物学和材料科学不断发展。",
        "古代丝绸之路不仅连接了东西方贸易网络，也让语言、宗教、艺术和技术在漫长时间里持续交流。",
        "海洋覆盖了地球表面的大部分区域，调节气候、影响降水，并为大量生物提供了复杂而稳定的栖息环境。",
    ],
    "numbers": [
        "请确认这张发票的编号是二零二六零五零五零一，金额为三百七十二元，付款账户尾号是九二八六。",
        "这趟列车计划上午九点四十五分出发，下午一点二十分到达，中间会在两个车站短暂停靠。",
        "模型一共训练了十二万个步骤，其中前两万个步骤用于学习率预热，最后一万步用于稳定收敛。",
        "这批数据包含一百六十条样本，预计总时长接近二十六分钟，每条样本目标长度约为十秒。",
        "请记录三个关键数值，分别是零点八五、一点二零和三点七六，并把它们写入今天的实验日志。",
        "今天是二零二六年五月五日，星期二，上午温度二十一度，下午可能会出现短时间阵雨。",
        "如果每个声线生成四十句文本，两个声线就是八十条样本，十六个声线则会扩展到六百四十条。",
        "服务器当前使用一张五零九零显卡，合成九分钟音频大约需要六分多钟，整体速度仍然可以接受。",
    ],
    "questions": [
        "如果同一句话在不同声线下都能保持清楚发音，我们是否可以更可靠地判断文本信息是否从声线中分离出来？",
        "当语速略有变化但说话人身份保持一致时，模型会把这种变化当成风格，还是错误地当成新的声线？",
        "如果验证集只保留没有见过的文本，而声线和条件仍然相同，我们能不能更准确地评估文本泛化能力？",
        "这条样本听起来是不是足够完整，句子中间的停顿是否自然，结尾有没有突然截断或者拖得太长？",
        "我们应该优先扩大声线数量，还是先把每个声线下的文本覆盖范围做得更加均衡和稳定？",
        "如果某个 prompt 的转写文本不准确，合成结果会不会明显变短，甚至破坏整个数据集的质量分布？",
        "这套蒸馏流程能否持续生成干净、可控、可扩展的数据，从而替代手工打标带来的巨大成本？",
        "当外部音素编码器和流匹配约束同时工作时，潜变量会不会更容易形成可生成的语义结构？",
    ],
    "hard": [
        "中英文混合文本会改变局部节奏，因此在构造训练样本时，需要控制英文片段的长度和出现位置。",
        "他说：“先别急着下结论，我们应该把合成音频、文本标注和模型输出放在一起检查。”",
        "如果括号里的补充说明读得太重，整句话的重点就会发生偏移，听起来也不像自然口语表达。",
        "在百分之九十五的情况下，简单规则已经足够稳定，但剩下的少数异常样本仍然需要单独记录。",
        "版本号从一点零升级到一点一，看起来只是小改动，但数据格式和评估脚本都必须保持兼容。",
        "人工智能、语音合成和自动语音识别经常出现在同一个系统里，但它们解决的问题并不完全相同。",
        "如果训练数据过于干净，模型可能缺少真实环境中的鲁棒性；如果数据太脏，语义结构又会被噪声干扰。",
        "我们需要的是可控的复杂度，而不是盲目堆叠变量，否则每个实验失败以后都很难判断真正原因。",
    ],
}


LONG10X_DOMAINS = {
    "daily": [
        ("项目组", "整理会议记录", "把负责人、截止时间和风险备注逐项写清楚", "第二天复盘时就不会遗漏关键事项"),
        ("运营同事", "核对订单信息", "确认金额、地址和联系方式都没有明显错误", "后续发货流程才能保持稳定"),
        ("值班工程师", "检查服务器状态", "记录显卡占用、磁盘空间和最近一次任务日志", "出现异常时可以迅速定位原因"),
        ("测试人员", "试听合成样本", "标记发音、停顿和音量变化比较突出的片段", "下一轮筛选就会更有依据"),
        ("数据负责人", "保存版本信息", "同步文本编号、声线编号和生成模型配置", "实验结果才容易被完整复现"),
        ("产品经理", "梳理用户反馈", "区分真实需求、偶发问题和暂时无法判断的现象", "团队讨论时可以少走弯路"),
        ("老师", "准备课堂材料", "把概念解释、例题步骤和课后练习放在同一份文档里", "学生预习时会更容易理解"),
        ("医生", "查看复诊记录", "确认症状变化、用药时间和检查结果是否一致", "后续判断才不会过于依赖单次描述"),
    ],
    "wiki": [
        ("长江流域", "覆盖多个气候区", "地形、降水和城市分布存在明显差异", "相关生态治理需要根据区域特点分别讨论"),
        ("太阳系", "包含多种天体", "行星、卫星、小行星和彗星共同构成复杂环境", "天文学研究因此需要长期观测数据"),
        ("印刷术", "降低了复制成本", "让书籍更快进入学校、商铺和普通家庭", "知识传播方式也随之发生深刻变化"),
        ("显微镜", "扩展了观察尺度", "帮助研究者看见细胞结构和微小材料缺陷", "现代医学与生物学因此获得重要工具"),
        ("海洋系统", "调节全球气候", "影响降水、洋流以及大量生物的栖息环境", "人类活动必须考虑长期生态影响"),
        ("城市交通", "连接不同功能区域", "道路、轨道和步行网络共同影响通勤效率", "规划方案需要兼顾成本和体验"),
        ("古代贸易路线", "促进远距离交流", "货物、语言、技术和艺术在漫长时间里相互影响", "历史研究常常需要跨地区证据"),
        ("机器学习", "依赖数据和目标函数", "样本质量、标签一致性和评估方式都会影响结果", "模型性能不能只看训练损失"),
    ],
    "numbers": [
        ("这批数据", "包含一百六十个文本组", "每个文本组会生成两个声线版本", "过滤以后预计还能保留二十分钟以上音频"),
        ("服务器任务", "计划连续运行三十分钟", "期间需要记录开始时间、结束时间和平均实时率", "方便估算下一批数据的生成成本"),
        ("发票记录", "编号是二零二六零五零五一二", "金额为三百七十二元，账户尾号是九二八六", "核对无误以后才能进入报销流程"),
        ("训练配置", "设置学习率为零点零零零一", "每四个小批次累积一次梯度", "这样可以在显存有限时保持稳定更新"),
        ("评估清单", "包含十项基础指标", "其中三项关注重建质量，四项关注语义读出", "剩余指标用于检查采样稳定性"),
        ("音频样本", "目标长度接近十秒", "允许范围暂时设定在八秒到十三秒之间", "超过范围的文本组会被整体剔除"),
        ("实验版本", "从一点一升级到一点二", "新增了长句文本池和按组过滤的质检清单", "后续报告需要引用准确路径"),
        ("声线目录", "目前只有两个有效提示音频", "如果扩展到十六个声线，样本数量会增长八倍", "因此必须提前做好自动质检"),
    ],
    "questions": [
        ("如果同一句话换成不同声线", "模型还能保持相同的文本结构吗", "我们需要比较音素读出和说话人读出是否互相干扰", "这样才知道约束是否真的起作用"),
        ("当样本长度接近十秒", "音素编码器会不会获得更稳定的上下文", "我们可以观察长句停顿、短语边界和语速估计是否更加可靠", "这比短句烟测更接近真实训练场景"),
        ("如果某个声线总是偏快", "是否应该单独降低它的合成速度", "否则同一文本组可能因为时长不一致被质检过滤", "数据利用率也会随之下降"),
        ("当验证集按文本组留出", "模型是否还能泛化到没有见过的句子", "这个问题比随机切分更重要", "因为随机切分容易让同一句文本泄漏到训练集中"),
        ("如果外部条件只包含文本", "生成模型是否会丢失说话人差异", "我们需要让声线标签和音素标签同时约束潜变量", "才能判断表示是否更可控"),
        ("当合成数据非常干净", "它会不会缺少真实录音里的复杂变化", "这个问题需要后续加入噪声和房间变量", "但当前阶段应该先把基础结构做稳"),
        ("如果一条样本被突然截断", "自动质检能不能及时发现", "时长范围、转写一致性和能量分布都可以作为筛选依据", "只依赖人工试听会很难扩展"),
        ("当声线数量扩大到十六个", "我们应该如何控制生成成本", "可以先做小批量校准，再批量合成完整清单", "这样比一次性全量生成更稳妥"),
    ],
    "hard": [
        ("中英文混合文本", "容易改变局部节奏", "如果英文片段过长，模型可能突然加速或者发音含混", "因此第一批长句仍然主要使用中文"),
        ("括号里的补充说明", "不应该读得过重", "否则听众会误以为它是句子的核心信息", "训练样本也会出现不自然的重音模式"),
        ("复杂变量设计", "需要遵循逐步扩展原则", "先固定声线和文本长度，再加入速度、情绪和背景变量", "失败时才容易定位真正原因"),
        ("潜变量约束", "不能只追求重建质量", "还要检查外部模态是否能读出文本、声线和语速", "否则下游流模型仍然可能难以采样"),
        ("自动转写结果", "可以帮助恢复提示音频文本", "但它仍然可能包含错字和缺失标点", "关键提示音频最好再经过人工复核"),
        ("过滤规则", "不能只删除单条样本", "如果同一文本的某个声线版本不合格，就应该整体剔除文本组", "这样交叉结构才不会被破坏"),
        ("真实数据", "通常更加自然但标注成本很高", "合成数据虽然略显规整，却能稳定控制声线、文本和时长", "两者以后可以形成互补关系"),
        ("流匹配训练", "对起点分布非常敏感", "如果潜变量无法从条件中生成，模型就会在采样阶段出现方差收缩", "这正是新架构需要解决的问题"),
    ],
}


def build_long10x_texts() -> dict[str, list[str]]:
    text_bank = {domain: list(texts) for domain, texts in LONG10_TEXTS.items()}
    safe_specs = {
        "daily": [
            ("整理会议记录", "核对负责人和截止时间", "标出需要继续跟进的事项", "第二天复盘时就不会遗漏关键问题"),
            ("核对订单信息", "确认金额和收货地址", "检查联系方式是否填写完整", "后续发货流程才能保持稳定"),
            ("检查服务器状态", "记录显卡占用和磁盘空间", "保存最近一次任务日志", "出现异常时可以迅速定位原因"),
            ("试听合成样本", "标记发音和停顿问题", "记录音量变化明显的片段", "下一轮筛选就会更有依据"),
            ("保存实验版本", "同步文本编号和声线编号", "记录生成模型和配置参数", "实验结果才容易被完整复现"),
            ("梳理用户反馈", "区分真实需求和偶发问题", "整理暂时无法判断的现象", "团队讨论时可以少走弯路"),
            ("准备课堂材料", "写清概念解释和例题步骤", "补充课后练习和参考答案", "学生预习时会更容易理解"),
            ("查看复诊记录", "确认症状变化和用药时间", "对照最近一次检查结果", "后续判断不会过于依赖单次描述"),
        ],
        "wiki": [
            ("介绍长江流域", "说明地形和气候差异", "补充城市分布和生态压力", "相关治理问题可以分区讨论"),
            ("介绍太阳系", "说明行星和卫星的关系", "补充小行星与彗星的运动特点", "听者能够理解更复杂的空间环境"),
            ("介绍印刷术", "说明书籍复制成本的下降", "补充学校和商铺中的传播变化", "知识普及过程会显得更加清楚"),
            ("介绍显微镜", "说明观察尺度的变化", "补充细胞结构和材料缺陷案例", "现代医学发展的线索更容易理解"),
            ("介绍海洋系统", "说明洋流和降水的联系", "补充生物栖息环境的变化", "气候调节作用会更加具体"),
            ("介绍城市交通", "说明道路和轨道的分工", "补充步行网络对通勤的影响", "规划方案的取舍会更加直观"),
            ("介绍古代贸易路线", "说明货物交换的范围", "补充语言和技术传播的证据", "历史交流过程会更加完整"),
            ("介绍机器学习", "说明样本质量的重要性", "补充标签一致性和评估方式", "模型表现就不会只看训练损失"),
        ],
        "numbers": [
            ("统计这批数据", "确认一百六十个文本组", "计算两个声线版本的样本数量", "过滤以后仍能保留较长音频"),
            ("记录服务器任务", "写下开始时间和结束时间", "计算平均实时率和总生成时长", "下一批数据成本就能提前估算"),
            ("核对发票记录", "确认编号和付款金额", "检查账户尾号和报销类别", "财务流程才不会出现反复修改"),
            ("检查训练配置", "确认学习率和累积步数", "记录批大小和片段长度", "显存有限时也能保持稳定更新"),
            ("整理评估清单", "区分重建质量和语义读出", "补充采样稳定性的检查项", "实验报告会更容易比较"),
            ("筛选音频样本", "确认目标长度接近十秒", "剔除低于八秒或高于十三秒的文本组", "交叉配对结构可以保持完整"),
            ("更新实验版本", "记录一点二版本的新增内容", "同步长句文本池和质检规则", "后续引用路径不会发生混乱"),
            ("规划声线目录", "统计当前有效提示音频", "估算扩展到十六个声线后的样本量", "自动质检必须提前准备好"),
        ],
        "questions": [
            ("评估同文本换声线", "比较音素读出是否稳定", "检查说话人读出是否互相干扰", "我们才能判断约束是否真的起作用"),
            ("评估十秒长句", "观察短语边界是否稳定", "检查停顿和语速估计是否可靠", "它会比短句烟测更接近真实场景"),
            ("评估偏快声线", "比较同一文本的生成时长", "尝试单独降低对应声线的速度", "数据利用率可能会明显提高"),
            ("评估文本组切分", "保证验证集没有见过相同句子", "保留不同声线下的完整配对", "泛化结果会比随机切分更可信"),
            ("评估外部条件", "同时输入文本和声线标签", "观察潜变量是否保留可读结构", "下游生成模型才更容易受控"),
            ("评估合成数据", "确认基础结构足够稳定", "再逐步加入噪声和房间变量", "实验失败时才容易定位原因"),
            ("评估异常截断", "检查时长范围和能量分布", "结合转写一致性筛掉坏样本", "数据扩展就不必完全依赖人工试听"),
            ("评估生成成本", "先做小批量声线校准", "再批量合成完整文本清单", "扩展到更多声线时会更加稳妥"),
        ],
        "hard": [
            ("处理中英文混合文本", "控制英文片段的长度", "观察局部节奏是否突然变化", "第一批长句仍然应该以中文为主"),
            ("处理括号补充说明", "避免把补充内容读得过重", "检查整句话的重点是否偏移", "训练样本会更接近自然口语"),
            ("处理复杂变量设计", "先固定声线和文本长度", "再逐步加入速度和情绪变量", "失败时才容易定位真正原因"),
            ("处理潜变量约束", "检查外部模态读出能力", "同时观察重建质量和流匹配损失", "新表示才不会只服务于重建"),
            ("处理自动转写结果", "复核关键提示音频文本", "修正可能出现的错字和缺失标点", "合成质量会更加稳定"),
            ("处理过滤规则", "按完整文本组进行剔除", "避免只删除某一个声线版本", "交叉结构就不会被破坏"),
            ("处理真实数据和合成数据", "比较自然度和标注成本", "保留可控变量和干净标签", "后续训练可以形成互补关系"),
            ("处理流匹配起点问题", "观察采样方差是否收缩", "检查条件是否足以预测潜变量方向", "新架构的价值才能被验证"),
        ],
    }
    templates = [
        "在{scene}时，我们会先{a}，再{b}，这样{result}。",
        "为了更稳定地{scene}，需要{a}，同时{b}，这样{result}。",
        "如果没有在{scene}时{a}，也没有{b}，就很难保证{result}。",
    ]
    for domain, specs in safe_specs.items():
        extra = []
        for scene, a, b, result in specs:
            a = a.removeprefix("先").removeprefix("再")
            b = b.removeprefix("先").removeprefix("再")
            for template in templates:
                text = template.format(scene=scene, a=a, b=b, result=result)
                extra.append(text.replace("同时同时", "同时"))
        text_bank.setdefault(domain, []).extend(extra)
    return text_bank


def build_long10align_texts() -> list[dict[str, str]]:
    """Build clustered long texts with controlled phoneme-neighbor structure."""
    specs = [
        (
            "align_daily",
            "meeting_records",
            "整理会议记录",
            ["核对负责人和截止时间", "核对负责人和完成时间", "核对负责人和交付时间", "核对整理人和截止时间", "核对记录人和截止时间"],
            ["标出需要继续跟进的事项", "标出需要持续跟进的事项", "标出需要后续跟进的事项", "标出需要重点跟进的事项", "标出需要立刻跟进的事项"],
            "第二天复盘时就不会遗漏关键问题",
        ),
        (
            "align_daily",
            "order_check",
            "核对订单信息",
            ["确认金额和收货地址", "确认金额和收件地址", "确认金额和配送地址", "确认价格和收货地址", "确认账单和收货地址"],
            ["检查联系方式是否填写完整", "检查联系号码是否填写完整", "检查联系电话是否填写完整", "检查联系地址是否填写完整", "检查联系信息是否填写完整"],
            "后续发货流程才能保持稳定",
        ),
        (
            "align_daily",
            "server_status",
            "检查服务器状态",
            ["记录显卡占用和磁盘空间", "记录显存占用和磁盘空间", "记录显卡负载和磁盘空间", "记录显卡占用和缓存空间", "记录显卡占用和剩余空间"],
            ["保存最近一次任务日志", "保存最近一轮任务日志", "保存最新一次任务日志", "保存最近一次训练日志", "保存最近一次合成日志"],
            "出现异常时可以迅速定位原因",
        ),
        (
            "align_speech",
            "sample_listen",
            "试听合成样本",
            ["标记发音和停顿问题", "标记发音和断句问题", "标记读音和停顿问题", "标记音量和停顿问题", "标记发音和节奏问题"],
            ["记录音量变化明显的片段", "记录能量变化明显的片段", "记录响度变化明显的片段", "记录音高变化明显的片段", "记录音色变化明显的片段"],
            "下一轮筛选就会更有依据",
        ),
        (
            "align_speech",
            "latent_probe",
            "分析潜变量结构",
            ["比较音素读出是否稳定", "比较声调读出是否稳定", "比较文本读出是否稳定", "比较音节读出是否稳定", "比较语义读出是否稳定"],
            ["检查说话人信息是否泄漏", "检查声线信息是否泄漏", "检查风格信息是否泄漏", "检查速度信息是否泄漏", "检查残差信息是否泄漏"],
            "我们才能判断约束是否真的起作用",
        ),
        (
            "align_speech",
            "long_sentence",
            "评估十秒长句",
            ["观察短语边界是否稳定", "观察句子边界是否稳定", "观察停顿边界是否稳定", "观察音素边界是否稳定", "观察语义边界是否稳定"],
            ["检查停顿和语速估计是否可靠", "检查停顿和音高估计是否可靠", "检查节奏和语速估计是否可靠", "检查停顿和时长估计是否可靠", "检查重音和语速估计是否可靠"],
            "它会比短句烟测更接近真实场景",
        ),
        (
            "align_numbers",
            "training_config",
            "检查训练配置",
            ["确认学习率和累积步数", "确认学习率和预热步数", "确认学习率和保存步数", "确认批大小和累积步数", "确认片段长和累积步数"],
            ["记录批大小和片段长度", "记录批大小和窗口长度", "记录批数量和片段长度", "记录显存占用和片段长度", "记录采样频率和片段长度"],
            "显存有限时也能保持稳定更新",
        ),
        (
            "align_numbers",
            "data_filter",
            "筛选音频样本",
            ["确认目标长度接近十秒", "确认目标时长接近十秒", "确认平均长度接近十秒", "确认目标长度接近九秒", "确认目标长度接近十二秒"],
            ["剔除低于八秒或高于十三秒的文本组", "剔除短于八秒或长于十三秒的文本组", "剔除低于七秒或高于十三秒的文本组", "剔除低于八秒或高于十二秒的文本组", "剔除低于八秒或高于十四秒的文本组"],
            "交叉配对结构可以保持完整",
        ),
        (
            "align_wiki",
            "river_region",
            "介绍长江流域",
            ["说明地形和气候差异", "说明地貌和气候差异", "说明地形和降水差异", "说明地形和温度差异", "说明区域和气候差异"],
            ["补充城市分布和生态压力", "补充城市分布和环境压力", "补充城镇分布和生态压力", "补充人口分布和生态压力", "补充城市分布和治理压力"],
            "相关治理问题可以分区讨论",
        ),
        (
            "align_wiki",
            "machine_learning",
            "介绍机器学习",
            ["说明样本质量的重要性", "说明标签质量的重要性", "说明数据质量的重要性", "说明样本数量的重要性", "说明样本分布的重要性"],
            ["补充标签一致性和评估方式", "补充标注一致性和评估方式", "补充标签完整性和评估方式", "补充标签一致性和验证方式", "补充标签一致性和测试方式"],
            "模型表现就不会只看训练损失",
        ),
    ]

    rows: list[dict[str, str]] = []
    for domain, cluster, scene, action_variants, detail_variants, result in specs:
        for variant_i, (action, detail) in enumerate(zip(action_variants, detail_variants)):
            if variant_i == 0:
                variant_type = "anchor"
                band = "same_cluster_anchor"
            elif variant_i in {1, 2}:
                variant_type = "near_substitution"
                band = "near"
            else:
                variant_type = "mid_substitution"
                band = "mid"
            text = f"在{scene}时，我们会先{action}，再{detail}，这样{result}。"
            rows.append(
                {
                    "domain": domain,
                    "text_index": f"{cluster}_{variant_i:02d}",
                    "text": text,
                    "phoneme_cluster_id": cluster,
                    "phoneme_variant_id": f"{cluster}_{variant_i:02d}",
                    "phoneme_variant_type": variant_type,
                    "expected_edit_distance_band": band,
                }
            )
    return rows


def build_long10scale_texts() -> list[dict[str, str]]:
    """Build a larger broad-coverage long-text pool for multi-hour synthesis."""
    def phrase(text: str) -> str:
        for prefix in ("先", "再", "并且", "同时"):
            text = text.removeprefix(prefix)
        return text

    def clean_text(text: str) -> str:
        replacements = {
            "先先": "先",
            "再再": "再",
            "并且并且": "并且",
            "同时同时": "同时",
            "并且再": "并且",
            "并且先": "并且",
        }
        for src, dst in replacements.items():
            text = text.replace(src, dst)
        return text

    rows: list[dict[str, str]] = []
    for domain, texts in build_long10x_texts().items():
        for index, text in enumerate(texts):
            rows.append(
                {
                    "domain": domain,
                    "text_index": f"base_{index:03d}",
                    "text": text,
                    "scale_source": "long10x_base",
                }
            )

    specs = {
        "scale_daily": {
            "scenes": ["整理会议材料", "核对订单信息", "检查设备状态", "准备课堂讲义", "处理用户反馈", "安排出差计划", "复查实验记录", "更新项目清单"],
            "actions": ["确认负责人和截止时间", "核对编号和关键备注", "记录当前状态和异常现象", "整理步骤和注意事项", "区分真实需求和偶发问题", "检查路线和住宿信息"],
            "details": ["补充需要继续跟进的事项", "保存便于复现的版本信息", "标出风险较高的环节", "把容易遗漏的细节单独列出", "同步给相关同事再次确认", "保留下一轮讨论所需证据"],
            "results": ["第二天复盘时就不会遗漏关键问题", "后续流程才能保持稳定", "团队讨论时会更容易达成一致", "出现异常时可以更快定位原因"],
        },
        "scale_speech": {
            "scenes": ["试听合成音频", "评估长句样本", "检查声线一致性", "分析潜变量结构", "准备蒸馏数据", "筛选异常片段", "比较重建结果", "记录语音指标"],
            "actions": ["标记发音和停顿问题", "观察短语边界是否稳定", "比较同文本下的声线差异", "检查音素读出是否可靠", "确认文本和音频是否一致", "统计音量和能量分布"],
            "details": ["记录变化明显的片段", "补充语速和停顿估计", "排除突然截断的样本", "保留跨说话人的完整配对", "加入近邻文本用于对齐", "检查说话人信息是否泄漏"],
            "results": ["下一轮训练会更容易判断问题来源", "外部约束才能获得更清楚的梯度", "模型结构是否有效会更容易验证", "数据扩展时不必完全依赖人工试听"],
        },
        "scale_wiki": {
            "scenes": ["介绍长江流域", "介绍太阳系结构", "介绍印刷术传播", "介绍城市交通", "介绍显微镜作用", "介绍海洋系统", "介绍机器学习", "介绍古代贸易路线"],
            "actions": ["说明背景和主要组成", "比较不同区域的差异", "梳理长期变化的过程", "解释关键概念的含义", "补充常见例子和证据", "区分主要因素和次要因素"],
            "details": ["强调数据和观察的重要性", "说明人类活动带来的影响", "补充历史材料中的线索", "把复杂关系拆成几个层次", "联系现实场景中的应用", "避免只依赖单一指标判断"],
            "results": ["听者能够形成更完整的理解", "相关问题可以根据场景分别讨论", "后续解释会显得更加清楚", "模型也能覆盖更稳定的知识类文本"],
        },
        "scale_numbers": {
            "scenes": ["核对发票记录", "统计数据规模", "检查训练配置", "记录服务器任务", "整理评估清单", "规划声线目录", "估算生成成本", "筛选音频样本"],
            "actions": ["确认编号和付款金额", "计算样本数量和总时长", "记录学习率和批大小", "写下开始时间和结束时间", "区分重建质量和语义读出", "估算扩展后的样本数量"],
            "details": ["补充账户尾号和报销类别", "比较过滤前后的保留比例", "检查显存占用和累积步数", "统计平均实时率和失败次数", "加入采样稳定性的检查项", "保留每个声线的质量备注"],
            "results": ["后续报告可以引用准确路径", "下一批数据成本就能提前估算", "显存有限时也能保持稳定更新", "实验之间会更容易公平比较"],
        },
        "scale_hard": {
            "scenes": ["处理中英文混合文本", "处理括号补充说明", "处理复杂变量设计", "处理自动转写结果", "处理流匹配训练", "处理真实数据和合成数据", "处理局部线性约束", "处理跨样本对齐"],
            "actions": ["控制困难片段的长度和位置", "避免补充内容读得过重", "先固定文本长度和声线条件", "复核可能出现的错字和缺失标点", "观察采样方差是否收缩", "比较自然度和标注成本"],
            "details": ["检查局部节奏是否突然变化", "确认整句话的重点没有偏移", "再逐步加入速度和环境变量", "保留干净标签和可控变量", "检查条件是否足以预测潜变量方向", "加入近邻和远邻的匹配关系"],
            "results": ["失败时才容易定位真正原因", "训练样本会更接近自然口语", "新表示才不会只服务于重建", "后续训练可以形成互补关系"],
        },
    }
    templates = [
        "在{scene}时，我们会先{action}，再{detail}，这样{result}。",
        "为了更稳定地{scene}，需要先{action}，并且{detail}，这样{result}。",
    ]
    for domain, spec in specs.items():
        item_index = 0
        for scene in spec["scenes"]:
            for action_i, action in enumerate(spec["actions"]):
                for detail_i, detail in enumerate(spec["details"]):
                    action = phrase(action)
                    detail = phrase(detail)
                    result = spec["results"][(action_i + detail_i) % len(spec["results"])]
                    template = templates[(action_i + detail_i) % len(templates)]
                    text = clean_text(template.format(scene=scene, action=action, detail=detail, result=result))
                    rows.append(
                        {
                            "domain": domain,
                            "text_index": f"scale_{item_index:04d}",
                            "text": text,
                            "scale_source": "long10scale_template",
                        }
                    )
                    item_index += 1
    return rows


DEFAULT_SPEAKERS = [
    {
        "speaker_id": "spk_zero_shot",
        "prompt_wav": "asset/zero_shot_prompt.wav",
        "prompt_text": PROMPT_TEXT,
        "speed": 1.0,
    },
    {
        "speaker_id": "spk_cross_lingual",
        "prompt_wav": "asset/cross_lingual_prompt.wav",
        "prompt_text": CROSS_LINGUAL_PROMPT_TEXT,
        "speed": 1.0,
    },
]


DEFAULT_STYLES = [
    {"style_id": "neutral_normal", "speed": 1.0},
    {"style_id": "neutral_fast", "speed": 1.08},
]

LONG10_STYLES = [
    {"style_id": "neutral_10s", "speed": 1.0},
]


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_texts(seed: int, max_texts: int, text_set: str, min_chars: int, max_chars: int) -> list[dict[str, Any]]:
    if text_set == "long10scale":
        rows = build_long10scale_texts()
    elif text_set == "long10align":
        rows = build_long10align_texts()
    elif text_set == "long10x":
        text_bank = build_long10x_texts()
        rows = []
        for domain, texts in text_bank.items():
            for index, text in enumerate(texts):
                rows.append({"domain": domain, "text_index": str(index), "text": text})
    elif text_set == "long10":
        text_bank = LONG10_TEXTS
        rows = []
        for domain, texts in text_bank.items():
            for index, text in enumerate(texts):
                rows.append({"domain": domain, "text_index": str(index), "text": text})
    else:
        text_bank = TEXTS
        rows = []
        for domain, texts in text_bank.items():
            for index, text in enumerate(texts):
                rows.append({"domain": domain, "text_index": str(index), "text": text})
    rows = [
        row for row in rows
        if (min_chars <= 0 or len(str(row["text"])) >= min_chars)
        and (max_chars <= 0 or len(str(row["text"])) <= max_chars)
    ]
    rng = random.Random(seed)
    rng.shuffle(rows)
    if max_texts > 0:
        rows = rows[:max_texts]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/tts/cosyvoice3/crossed_zh_v0_texts.jsonl"))
    parser.add_argument("--speakers-json", type=Path, default=None)
    parser.add_argument("--styles-json", type=Path, default=None)
    parser.add_argument("--text-set", choices=["short", "long10", "long10x", "long10align", "long10scale"], default="short")
    parser.add_argument("--version", default="crosszh_v0")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--max-texts", type=int, default=24)
    parser.add_argument("--min-chars", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=8)
    args = parser.parse_args()

    speakers = read_json(args.speakers_json) if args.speakers_json else DEFAULT_SPEAKERS
    default_styles = LONG10_STYLES if args.text_set in {"long10", "long10x", "long10align", "long10scale"} else DEFAULT_STYLES
    styles = read_json(args.styles_json) if args.styles_json else default_styles
    if not isinstance(speakers, list) or not speakers:
        raise ValueError("speakers must be a non-empty JSON list")
    if not isinstance(styles, list) or not styles:
        raise ValueError("styles must be a non-empty JSON list")

    text_rows = flatten_texts(args.seed, args.max_texts, args.text_set, args.min_chars, args.max_chars)
    rows = []
    for text_i, text_row in enumerate(text_rows):
        text_id = f"txt_{text_i:04d}_{text_row['domain']}_{text_row['text_index']}"
        split = "val" if args.val_every > 0 and text_i % args.val_every == 0 else "train"
        for speaker in speakers:
            for style in styles:
                speaker_id = str(speaker["speaker_id"])
                style_id = str(style["style_id"])
                speaker_speed = float(speaker.get("speed", 1.0))
                style_speed = float(style.get("speed", 1.0))
                utt_id = f"{args.version}_{len(rows):05d}_{speaker_id}_{style_id}"
                row = {
                        "id": utt_id,
                        "text": text_row["text"],
                        "language": "zh",
                        "text_set": args.text_set,
                        "domain": text_row["domain"],
                        "text_id": text_id,
                        "same_text_group": text_id,
                        "speaker_id": speaker_id,
                        "same_speaker_group": speaker_id,
                        "style_id": style_id,
                        "style": style_id,
                        "condition_id": f"{speaker_id}__{style_id}",
                        "speed": speaker_speed * style_speed,
                        "speaker_speed": speaker_speed,
                        "style_speed": style_speed,
                        "prompt_wav": str(speaker["prompt_wav"]),
                        "prompt_text": str(speaker.get("prompt_text", PROMPT_TEXT)),
                        "split": split,
                    }
                for key in ("phoneme_cluster_id", "phoneme_variant_id", "phoneme_variant_type", "expected_edit_distance_band", "scale_source"):
                    if key in text_row:
                        row[key] = text_row[key]
                rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "out": str(args.out),
        "items": len(rows),
        "texts": len(text_rows),
        "speakers": len(speakers),
        "styles": len(styles),
        "train": sum(1 for row in rows if row["split"] == "train"),
        "val": sum(1 for row in rows if row["split"] == "val"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
