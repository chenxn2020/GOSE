import os
import re

import numpy as np

from transformers.utils import logging
import sys

logger = logging.get_logger(__name__)
# logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stdout)],
#     )
# logger.setLevel(logging.INFO)

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))

def calc_cross(pred_links, gt_links):
    tp_links = pred_links & gt_links #预测对的links
    fp_links = pred_links - gt_links #预测错的links
    total_links = pred_links
    cross2total = count_cross(fp_links, total_links)
    cross2tp = count_cross(fp_links, tp_links)
    cross2fp = count_cross(fp_links, fp_links)
    return cross2total, cross2tp, cross2fp
        
def count_cross(false_links, total_links):
    count = 0
    for false_link in false_links:
        is_cross = False
        for link in total_links:
            if false_link == link:
                continue
            p1 = Point(false_link[0])
            p2 = Point(false_link[1])
            p3 = Point(link[0])
            p4 = Point(link[1])
            if IsIntersec(p1,p2,p3,p4):
                is_cross = True
                break      
        if is_cross:
            count += 1
    return count            
            
            # D = IsIntersec()
#Python3.6
class Point(): #定义类
    def __init__(self, point):
        x, y = point
        self.x=x
        self.y=y   

def cross(p1,p2,p3):#跨立实验
    x1=p2.x-p1.x
    y1=p2.y-p1.y
    x2=p3.x-p1.x
    y2=p3.y-p1.y
    return x1*y2-x2*y1     

def IsIntersec(p1,p2,p3,p4): #判断两线段是否相交

    #快速排斥，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if(max(p1.x,p2.x)>=min(p3.x,p4.x)    #矩形1最右端大于矩形2最左端
    and max(p3.x,p4.x)>=min(p1.x,p2.x)   #矩形2最右端大于矩形最左端
    and max(p1.y,p2.y)>=min(p3.y,p4.y)   #矩形1最高端大于矩形最低端
    and max(p3.y,p4.y)>=min(p1.y,p2.y)): #矩形2最高端大于矩形最低端

    #若通过快速排斥则进行跨立实验, 遵循严格跨立，即严格交叉准则
        if(cross(p1,p2,p3)*cross(p1,p2,p4)<0
           and cross(p3,p4,p1)*cross(p3,p4,p2)<0):
            D=True
        else:
            D=False
    else:
        D=False
    return D



def re_score(pred_relations, gt_relations, mode="strict"):
    """Evaluate RE predictions

    Args:
        pred_relations (list) :  list of list of predicted relations (several relations in each sentence)
        gt_relations (list) :    list of list of ground truth relations

            rel = { "head": (start_idx (inclusive), end_idx (exclusive)),
                    "tail": (start_idx (inclusive), end_idx (exclusive)),
                    "head_type": ent_type,
                    "tail_type": ent_type,
                    "type": rel_type}

        vocab (Vocab) :         dataset vocabulary
        mode (str) :            in 'strict' or 'boundaries'"""

    assert mode in ["strict", "boundaries"]
    relation_types = [v for v in [0, 1] if not v == 0]
    scores = {rel: {"tp": 0, "fp": 0, "fn": 0} for rel in relation_types + ["ALL"]}
    cross_count = {"cross2total": 0, "cross2tp": 0, "cross2fp": 0}
    # Count GT relations and Predicted relations
    n_sents = len(gt_relations)
    n_rels = sum([len([rel for rel in sent]) for sent in gt_relations]) #ground-truth中links的总数
    n_found = sum([len([rel for rel in sent]) for sent in pred_relations]) #预测的links的总数
    print(f"gt_rels:{n_rels}, pred_rels:{n_found}\n")
    # Count TP, FP and FN per type
    for pred_sent, gt_sent in zip(pred_relations, gt_relations):
        #遍历每份文档
        for rel_type in relation_types:
            # strict mode takes argument types into account
            if mode == "strict":
                pred_rels = {
                    (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
                    for rel in pred_sent
                    if rel["type"] == rel_type
                }
                gt_rels = {
                    (rel["head"], rel["head_type"], rel["tail"], rel["tail_type"])
                    for rel in gt_sent
                    if rel["type"] == rel_type
                }

            # boundaries mode only takes argument spans into account
            elif mode == "boundaries":
                #---统计每份文档中的links
                pred_rels = {(rel["head"], rel["tail"]) for rel in pred_sent if rel["type"] == rel_type}
                gt_rels = {(rel["head"], rel["tail"]) for rel in gt_sent if rel["type"] == rel_type}
                pred_links = {rel["link"] for rel in pred_sent if rel["type"] == rel_type}
                gt_links = {rel["link"] for rel in gt_sent if rel["type"] == rel_type}
            scores[rel_type]["tp"] += len(pred_rels & gt_rels)
            scores[rel_type]["fp"] += len(pred_rels - gt_rels)
            scores[rel_type]["fn"] += len(gt_rels - pred_rels)
            #---计算各种设置下交叉link的个数
            cross2total, cross2tp, cross2fp = calc_cross(pred_links, gt_links)
            cross_count["cross2total"] += cross2total
            cross_count["cross2tp"] += cross2tp
            cross_count["cross2fp"] += cross2fp
            #------------------------
            
    #---tp+fp == n_found
    # Compute per entity Precision / Recall / F1
    for rel_type in scores.keys():
        if scores[rel_type]["tp"]:
            scores[rel_type]["p"] = scores[rel_type]["tp"] / (scores[rel_type]["fp"] + scores[rel_type]["tp"])
            scores[rel_type]["r"] = scores[rel_type]["tp"] / (scores[rel_type]["fn"] + scores[rel_type]["tp"])
        else:
            scores[rel_type]["p"], scores[rel_type]["r"] = 0, 0

        if not scores[rel_type]["p"] + scores[rel_type]["r"] == 0:
            scores[rel_type]["f1"] = (
                2 * scores[rel_type]["p"] * scores[rel_type]["r"] / (scores[rel_type]["p"] + scores[rel_type]["r"])
            )
        else:
            scores[rel_type]["f1"] = 0

    # Compute micro F1 Scores
    tp = sum([scores[rel_type]["tp"] for rel_type in relation_types])
    fp = sum([scores[rel_type]["fp"] for rel_type in relation_types])
    fn = sum([scores[rel_type]["fn"] for rel_type in relation_types])
    if tp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

    else:
        precision, recall, f1 = 0, 0, 0

    scores["ALL"]["p"] = precision
    scores["ALL"]["r"] = recall
    scores["ALL"]["f1"] = f1
    scores["ALL"]["tp"] = tp
    scores["ALL"]["fp"] = fp
    scores["ALL"]["fn"] = fn
    #----计算link的交叉率
    if n_found == 0:
        scores["ALL"]["cross_total_ratio"] = 0
    else:
        scores["ALL"]["cross_total_ratio"] = cross_count["cross2total"] / n_found

    # Compute Macro F1 Scores
    scores["ALL"]["Macro_f1"] = np.mean([scores[ent_type]["f1"] for ent_type in relation_types])
    scores["ALL"]["Macro_p"] = np.mean([scores[ent_type]["p"] for ent_type in relation_types])
    scores["ALL"]["Macro_r"] = np.mean([scores[ent_type]["r"] for ent_type in relation_types])

    logger.info(f"RE Evaluation in *** {mode.upper()} *** mode")

    logger.info(
        "processed {} sentences with {} relations; found: {} relations; correct: {}.".format(
            n_sents, n_rels, n_found, tp
        )
    )
    logger.info(
        "\tALL\t TP: {};\tFP: {};\tFN: {}".format(scores["ALL"]["tp"], scores["ALL"]["fp"], scores["ALL"]["fn"])
    )
    logger.info("\t\t(m avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (micro)".format(precision, recall, f1))
    logger.info(
        "\t\t(M avg): precision: {:.2f};\trecall: {:.2f};\tf1: {:.2f} (Macro)\n".format(
            scores["ALL"]["Macro_p"], scores["ALL"]["Macro_r"], scores["ALL"]["Macro_f1"]
        )
    )

    for rel_type in relation_types:
        logger.info(
            "\t{}: \tTP: {};\tFP: {};\tFN: {};\tprecision: {:.2f};\trecall: {:.2f};\tf1: {:.2f};\t{}".format(
                rel_type,
                scores[rel_type]["tp"],
                scores[rel_type]["fp"],
                scores[rel_type]["fn"],
                scores[rel_type]["p"],
                scores[rel_type]["r"],
                scores[rel_type]["f1"],
                scores[rel_type]["tp"] + scores[rel_type]["fp"],
            )
        )

    return scores
