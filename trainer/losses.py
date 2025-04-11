import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss(model, ref_model, inputs, loss_type, beta=0.1):
    # forget_loss
    if 'GA1' in loss_type:
        forget_loss = ga1_loss(model, inputs)
    elif 'GA2' in loss_type:
        forget_loss = ga2_loss(model, inputs)
    elif 'GA3' in loss_type:
        forget_loss = ga3_loss(model, inputs)
    elif 'GA5' in loss_type:
        forget_loss = ga5_loss(model,inputs)
    elif 'GA6' in loss_type:
        forget_loss = ga6_loss(model,inputs)
    elif 'FINETUNE' in loss_type:
        forget_loss = finetune_loss(model,inputs)
    elif 'RETRAIN' in loss_type:
        forget_loss = finetune_loss(model,inputs)
    elif 'NPO1' in loss_type:
        forget_loss = npo1_loss(model, ref_model, inputs, beta=beta)
    elif 'NPO2' in loss_type:
        forget_loss = npo2_loss(model, ref_model, inputs, beta=beta)
    elif 'NPO3' in loss_type:
        forget_loss = npo3_loss(model, ref_model, inputs, beta=beta)
    elif 'DPO1' in loss_type:
        forget_loss = dpo1_loss(model, ref_model, inputs, beta=beta)
    elif 'DPO2' in loss_type:
        forget_loss = dpo2_loss(model, ref_model, inputs, beta=beta)
    elif 'ME' in loss_type:
        forget_loss = me_loss(model, inputs)
    elif 'IDK1' in loss_type:
        forget_loss = idk1_loss(model, inputs)
    elif 'IDK2' in loss_type:
        forget_loss = idk2_loss(model, inputs)
    elif 'IDK3' in loss_type:
        forget_loss = idk3_loss(model, inputs)
    elif 'SDK' in loss_type:
        forget_loss = sdk_loss(model, inputs)

    # regularization_loss
    if 'GD1' in loss_type:
        regularization_loss = gd1_loss(model, inputs)
    elif 'GD2' in loss_type:
        regularization_loss = gd2_loss(model, inputs)
    elif 'GD3' in loss_type:
        regularization_loss = gd3_loss(model, inputs)
    elif 'KL1' in loss_type:
        regularization_loss = kl1_loss(model, ref_model, inputs)
    elif 'KL2' in loss_type:
        regularization_loss = kl2_loss(model, ref_model, inputs)
    elif 'KL3' in loss_type:
        regularization_loss = kl3_loss(model, ref_model, inputs)
    elif 'AP' in loss_type:
        regularization_loss = ap_loss(model, inputs, beta=beta)
    else:
        regularization_loss = 0

    return forget_loss, regularization_loss


def ga1_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss
def ga2_loss(model, inputs):
    forget_inputs = inputs[4]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss
def ga3_loss(model, inputs):
    forget_inputs = inputs[6]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss
def ga5_loss(model, inputs):
    forget_inputs = inputs[11]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss
def ga6_loss(model, inputs):
    forget_inputs = inputs[12]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = -1 * outputs.loss
    return loss
def finetune_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    loss = +1 * outputs.loss
    return loss


# def sga_loss(model, inputs):
#     forget_inputs = inputs[0]
#     input_ids, labels, attention_mask = forget_inputs

#     with torch.no_grad():  # 그래디언트 계산 방지
#         # 모델을 배치 전체에 대해 한 번 실행
#         outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
#         logits = outputs.logits  # (batch_size, seq_len, vocab_size)

#         # 확률 계산 (Softmax 적용)
#         probs = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)

#         # 각 위치에서 정답 레이블에 해당하는 확률 가져오기
#         batch_indices = torch.arange(probs.shape[0]).unsqueeze(1)  # (batch_size, 1)
#         seq_indices = torch.arange(probs.shape[1])  # (seq_len,)
#         selected_probs = probs[batch_indices, seq_indices, labels]  # (batch_size, seq_len)

#         # -100 (패딩) 제외한 평균 확률 계산
#         valid_mask = labels != -100  # 패딩이 아닌 부분만 True
#         avg_probs = torch.sum(selected_probs * valid_mask, dim=1) / valid_mask.sum(dim=1)  # (batch_size,)

#     return avg_probs.cpu().numpy().tolist()  # 각 샘플별 평균 확률 리스트 반환


def npo1_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels,
                                attention_mask=attention_mask)
        loss_ref = get_batch_loss(ref_outputs.logits, labels)

    neg_log_ratios = loss_current - loss_ref
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    return loss
def npo2_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs = inputs[4]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels,
                                attention_mask=attention_mask)
        loss_ref = get_batch_loss(ref_outputs.logits, labels)

    neg_log_ratios = loss_current - loss_ref
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    return loss
def npo3_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs = inputs[6]
    input_ids, labels, attention_mask = forget_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss_current = get_batch_loss(outputs.logits, labels)

    with torch.no_grad():
        ref_outputs = ref_model(input_ids, labels=labels,
                                attention_mask=attention_mask)
        loss_ref = get_batch_loss(ref_outputs.logits, labels)

    neg_log_ratios = loss_current - loss_ref
    loss = - F.logsigmoid(beta * neg_log_ratios).mean() * 2 / beta

    return loss

def idk1_loss(model, inputs):
    forget_idk_inputs = inputs[2]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss
def idk2_loss(model, inputs):
    forget_idk_inputs = inputs[8]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss
def idk3_loss(model, inputs):
    forget_idk_inputs = inputs[10]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def sdk_loss(model, inputs):
    forget_idk_inputs = inputs[13]
    input_ids, labels, attention_mask = forget_idk_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss

def dpo1_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs, forget_idk_inputs = inputs[0], inputs[2]
    forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
    idk_input_ids, idk_labels, idk_attention_mask = forget_idk_inputs

    idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
    forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
    idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
    forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

    with torch.no_grad():
        idk_outputs_ref = ref_model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
        forget_outputs_ref = ref_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
        idk_loss_ref = -1 * get_batch_loss(idk_outputs_ref.logits, idk_labels)
        forget_loss_ref = -1 * get_batch_loss(forget_outputs_ref.logits, forget_labels)

    pi_logratios = idk_loss_current - forget_loss_current
    ref_logratios = idk_loss_ref - forget_loss_ref
    loss = - F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean() * 2 / beta
    return loss

def dpo2_loss(model, ref_model, inputs, beta=0.1):
    forget_inputs, forget_idk_inputs = inputs[4], inputs[8]
    forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
    idk_input_ids, idk_labels, idk_attention_mask = forget_idk_inputs

    idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
    forget_outputs = model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
    idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
    forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

    with torch.no_grad():
        idk_outputs_ref = ref_model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
        forget_outputs_ref = ref_model(forget_input_ids, labels=forget_labels, attention_mask=forget_attention_mask)
        idk_loss_ref = -1 * get_batch_loss(idk_outputs_ref.logits, idk_labels)
        forget_loss_ref = -1 * get_batch_loss(forget_outputs_ref.logits, forget_labels)

    pi_logratios = idk_loss_current - forget_loss_current
    ref_logratios = idk_loss_ref - forget_loss_ref
    loss = - F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean() * 2 / beta
    return loss

# Regularization Loss: AP
def ap_loss(model, inputs, beta=0.1):
    retain_inputs, retain_idk_inputs = inputs[1], inputs[3]
    retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
    retain_idk_input_ids, retain_idk_labels, retain_idk_attention_mask = retain_idk_inputs

    outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    idk_outputs = model(retain_idk_input_ids, labels=retain_idk_labels, attention_mask=retain_idk_attention_mask)

    loss = get_batch_loss(outputs.logits, retain_labels)
    loss_idk = get_batch_loss(idk_outputs.logits, retain_idk_labels)

    neg_log_ratios = loss_idk - loss

    loss = - F.logsigmoid(beta * neg_log_ratios).mean() / beta

    return loss


# Regularization Loss: KL
def kl1_loss(model, ref_model, inputs):
    retain_inputs = inputs[1]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    logits = outputs.logits.detach()
    probs = F.log_softmax(outputs.logits, dim=-1).view(-1, logits.shape[-1])

    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)
    ref_probs = F.log_softmax(outputs_ref.logits, dim=-1).view(-1, outputs_ref.logits.shape[-1])

    loss = nn.functional.kl_div(
        probs, ref_probs, reduction='batchmean', log_target=True)

    return loss
# Regularization Loss: KL
def kl2_loss(model, ref_model, inputs):
    retain_inputs = inputs[5]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    probs = F.log_softmax(outputs.logits, dim=-1).view(-1, outputs.logits.shape[-1])

    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)
    ref_probs = F.log_softmax(outputs_ref.logits, dim=-1).view(-1, outputs_ref.logits.shape[-1])

    loss = nn.functional.kl_div(
        probs, ref_probs, reduction='batchmean', log_target=True)

    return loss
# Regularization Loss: KL
def kl3_loss(model, ref_model, inputs):
    retain_inputs = inputs[7]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    probs = F.log_softmax(outputs.logits, dim=-1).view(-1, outputs.logits.shape[-1])

    with torch.no_grad():
        outputs_ref = ref_model(input_ids, labels=labels, attention_mask=attention_mask)
    ref_probs = F.log_softmax(outputs_ref.logits, dim=-1).view(-1, outputs_ref.logits.shape[-1])

    loss = nn.functional.kl_div(
        probs, ref_probs, reduction='batchmean', log_target=True)

    return loss

def mismatch_loss(model, inputs):
    mismatch_inputs = inputs[4]
    input_ids, labels, attention_mask = mismatch_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)

    loss = outputs.loss
    return loss


# Regularization Loss: GD
def gd1_loss(model, inputs):
    retain_inputs = inputs[1]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss
def gd2_loss(model, inputs):
    retain_inputs = inputs[5]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss
def gd3_loss(model, inputs):
    retain_inputs = inputs[7]
    input_ids, labels, attention_mask = retain_inputs

    outputs = model(input_ids, labels=labels,
                    attention_mask=attention_mask)
    loss = outputs.loss
    return loss


def get_batch_loss(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss


def me_loss(model, inputs):
    forget_inputs = inputs[0]
    input_ids, labels, attention_mask = forget_inputs
    outputs = model(input_ids, labels=None, attention_mask=attention_mask)
    loss = get_me_loss(outputs.logits, labels)

    return loss


def get_me_loss(logits, labels):
    num_labels = logits.shape[-1]

    assert logits.shape[:-1] == labels.shape, "Logits and labels must have compatible shapes."

    # Adjust logits and labels to exclude the last token
    labels = labels[:, 1:].clone()  # (bs, seq_len - 1)
    logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)

    soft_outputs = F.softmax(logits, dim=-1).view(-1, num_labels)  # (bs*seq_len, vocab_size)
    uniform_dist = torch.full_like(soft_outputs, 1.0 / num_labels).to(logits.device)  # (bs*seq_len, vocab_size)

    loss_mask = (labels != -100).view(-1)  # (bs*(seq_len - 1))

    kl_div = F.kl_div((soft_outputs + 1e-12).log(), uniform_dist, reduction='none').sum(-1)  # (bs*(seq_len - 1))

    masked_kl_div = kl_div * loss_mask  # (bs*(seq_len - 1))
    loss = masked_kl_div.sum() / loss_mask.sum()

    return loss
