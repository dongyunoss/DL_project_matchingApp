"""공용 추론 루프 (평가/보정/예측에서 재사용)."""
import torch


@torch.no_grad()
def run_inference(model, loader, device, amp: bool = False, tta: bool = False, return_embeddings: bool = False):
    """loader의 배치 (x, label_or_index, ...)에 대해 logits(와 임베딩)을 모은다.

    TTA: 원본과 좌우 반전 이미지의 logits/임베딩 평균.
    반환: dict(logits=(N,K), labels=(N,), embeddings=(N,D) or None)
    """
    model.eval()
    use_amp = amp and device.type == "cuda"
    logits_all, labels_all, emb_all = [], [], []
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits, emb = model.forward_with_features(x)
            if tta:
                l2, e2 = model.forward_with_features(torch.flip(x, dims=[3]))
                logits, emb = (logits + l2) / 2, (emb + e2) / 2
        logits_all.append(logits.float().cpu())
        labels_all.append(torch.as_tensor(y))
        if return_embeddings:
            emb_all.append(emb.float().cpu())
    return {
        "logits": torch.cat(logits_all) if logits_all else torch.empty(0),
        "labels": torch.cat(labels_all) if labels_all else torch.empty(0, dtype=torch.long),
        "embeddings": torch.cat(emb_all) if return_embeddings and emb_all else None,
    }
