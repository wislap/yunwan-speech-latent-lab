#!/usr/bin/env python3
"""Create a small controlled Chinese text manifest for CosyVoice3 distillation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/tts/cosyvoice3/seed_zh_v0_texts.jsonl"))
    parser.add_argument("--speaker-id", default="cv3_default_zh_female")
    parser.add_argument("--prompt-wav", default="asset/zero_shot_prompt.wav")
    parser.add_argument("--prompt-text", default=PROMPT_TEXT)
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    speeds = [0.92, 1.0, 1.08]
    speed_variant_index = 0
    for repeat in range(args.repeat):
        domains = list(TEXTS.items())
        rng.shuffle(domains)
        for domain, texts in domains:
            shuffled = list(texts)
            rng.shuffle(shuffled)
            for text_index, text in enumerate(shuffled):
                speed = 1.0
                style = "neutral"
                if repeat == args.repeat - 1 and text_index % 4 == 0:
                    speed = speeds[speed_variant_index % len(speeds)]
                    speed_variant_index += 1
                    style = f"neutral_speed_{speed:g}"
                utt_id = f"seedzh_v0_{len(rows):05d}"
                rows.append(
                    {
                        "id": utt_id,
                        "text": text,
                        "language": "zh",
                        "domain": domain,
                        "style": style,
                        "speed": speed,
                        "speaker_id": args.speaker_id,
                        "prompt_wav": args.prompt_wav,
                        "prompt_text": args.prompt_text,
                        "split": "train" if len(rows) % 10 else "val",
                    }
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"out": str(args.out), "items": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
