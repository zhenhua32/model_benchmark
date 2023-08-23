import json
import os

from tqdm import tqdm


data_dir = r"G:\dataset\text_classify\tnews_public\raw"


def load_label():
    """
    加载标签字典
    """
    label_file = os.path.join(data_dir, "labels.json")
    label2id = dict()
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            line = json.loads(line)
            label_id = line["label"]
            label = line["label_desc"]
            label2id[label] = label_id

    return label2id


def predict_one_line(line: str, apply_func: callable, label2id: dict) -> str:
    """
    预测一行数据
    line: 输入的文本, 没有转换成 json 的
    apply_func: 模型预测函数, 应该接受一个文本, 然后预测出一个标签
    label2id: 标签字典, 从标签 => id
    """
    line = json.loads(line)
    id = line["id"]
    sentence = line["sentence"]

    label = apply_func(sentence)
    label_id = label2id[label]

    return json.dumps({"id": id, "label": label_id, "label_desc": label}, ensure_ascii=False)


def predict_file(input_file: str, output_file: str, apply_func: callable):
    """
    在整个文件上进行预测
    """
    label2id = load_label()
    with open(input_file, "r", encoding="utf-8") as f:
        lines = []
        for line in tqdm(f):
            lines.append(predict_one_line(line, apply_func, label2id))
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
