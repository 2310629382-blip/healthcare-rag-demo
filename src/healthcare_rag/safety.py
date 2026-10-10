"""医疗场景的安全兜底：紧急症状识别 + 免责声明 + 范围外拒答话术。

Demo 不等于医疗建议。这里用最简单的规则做一层护栏：
1. 命中紧急症状关键词 → 直接建议拨打 120 / 立即就医，不输出诊疗方案；
2. 涉及具体用药剂量 → 提示以医生和说明书为准；
3. 知识库没有依据 → 明确说明「不知道」，而不是让大模型编。
"""

from __future__ import annotations

from dataclasses import dataclass

DISCLAIMER = (
    "本 Demo 仅用于技术学习与知识科普，内容整理自公开健康科普资料，"
    "不能替代医生面诊、诊断和处方。如有不适请及时就医。"
)

# 需要立即就医的典型危险信号（社区场景常见）
EMERGENCY_KEYWORDS = [
    "胸痛", "胸闷痛", "心绞痛", "胸口压榨", "一侧肢体无力", "半边身子", "嘴歪",
    "说话不清", "言语不清", "意识不清", "昏迷", "抽搐", "呼吸困难", "喘不上气",
    "窒息", "噎住", "大出血", "呕血", "咯血", "便血", "黑便", "剧烈头痛",
    "血压200", "血压 200", "血糖昏迷", "低血糖昏迷", "跌倒后剧痛", "摔后不能动",
    "骨折", "自杀", "不想活", "轻生",
]

EMERGENCY_MESSAGE = (
    "【紧急提醒】你描述的情况可能是需要立即处理的急诊信号：**请马上拨打 120 或让家人陪同前往最近的急诊**，"
    "不要自行加药、停药或在家观察。以下内容仅为知识库中的一般性科普，不能用于急救判断。"
)

# 与「用药」直接相关的表述（"吃多少盐"这种饮食提问不应被误判为用药问题）
DOSE_KEYWORDS = ["剂量", "加量", "减量", "停药", "换药", "几片", "自行用药", "加倍", "补服"]
MEDICATION_ACTIONS = ["吃", "服", "用", "停", "加", "减", "换"]


@dataclass
class SafetyHint:
    level: str          # "emergency" / "caution" / None
    message: str


def check_question(question: str) -> SafetyHint:
    """对用户提问做规则安全检查。"""
    text = (question or "").strip()
    if not text:
        return SafetyHint(level="", message="")

    if any(keyword in text for keyword in EMERGENCY_KEYWORDS):
        return SafetyHint(level="emergency", message=EMERGENCY_MESSAGE)

    mentions_medication = "药" in text
    dose_like = any(keyword in text for keyword in DOSE_KEYWORDS) or (
        mentions_medication and any(action in text for action in MEDICATION_ACTIONS)
    )
    if dose_like:
        return SafetyHint(
            level="caution",
            message="【用药提醒】具体用药方案请以医生处方和药品说明书为准，下面是知识库中的一般性提醒。",
        )

    return SafetyHint(level="", message="")


def build_refusal(question: str, threshold: float) -> str:
    """知识库中没有依据时的回答。"""
    return (
        f"我在当前知识库里没有找到能支撑「{question}」的可靠资料"
        f"（最高相关度低于阈值 {threshold:.2f}）。\n\n"
        "建议：\n"
        "1. 换一种问法，或补充更具体的场景（例如年龄、是否有基础疾病、想了解饮食还是运动）；\n"
        "2. 直接咨询社区卫生服务中心的全科医生或家庭医生，他们能给出针对性建议；\n"
        "3. 如果身体已经出现不适，不要依赖问答结果，请及时就医。\n\n"
        f"> {DISCLAIMER}"
    )