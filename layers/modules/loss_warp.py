import torch
import torch.nn.functional as F

def logloss(Ptrue, Pred, szs, eps=1e-10):
    b, h, w, ch = szs
    Pred = torch.clamp(Pred, eps, 1.0)
    Pred = -torch.log(Pred)
    Pred = Pred * Ptrue
    Pred = Pred.view(b, h * w * ch)
    Pred = Pred.sum(dim=1)
    return Pred

def l1(true, pred, szs):
    b, h, w, ch = szs
    res = (true - pred).view(b, h * w * ch)
    res = torch.abs(res)
    res = res.sum(dim=1)
    return res

def loss(Ytrue, Ypred):
    b, h, w = Ytrue.size(0), Ytrue.size(1), Ytrue.size(2)

    obj_probs_true = Ytrue[..., 0]
    obj_probs_pred = Ypred[..., 0]

    non_obj_probs_true = 1.0 - Ytrue[..., 0]
    non_obj_probs_pred = Ypred[..., 1]

    affine_pred = Ypred[..., 2:]
    pts_true = Ytrue[..., 1:]

    affinex = torch.stack([torch.clamp(affine_pred[..., 0], min=0.0), affine_pred[..., 1], affine_pred[..., 2]], dim=-1)
    affiney = torch.stack([affine_pred[..., 3], torch.clamp(affine_pred[..., 4], min=0.0), affine_pred[..., 5]], dim=-1)

    v = 0.5
    base = torch.tensor([[-v, -v, 1.0, v, -v, 1.0, v, v, 1.0, -v, v, 1.0]])
    base = base.view(1, 1, 1, -1)
    base = base.expand(b, h, w, -1).to(Ytrue.device)
    pts = torch.zeros(b, h, w, 0, device=Ytrue.device)

    for i in range(0, 12, 3):
        row = base[..., i:(i+3)]

        ptsx = (affinex * row).sum(dim=-1)
        print(affinex.shape, row.shape, ptsx.shape)
        ptsy = (affiney * row).sum(dim=-1)

        pts_xy = torch.stack([ptsx, ptsy], dim=-1)
        pts = torch.cat([pts, pts_xy], dim=-1)
    print(pts.shape)

    flags = obj_probs_true.view(b, h, w, 1)
    res = 1.0 * l1(pts_true * flags, pts * flags, (b, h, w, 4 * 2))
    return res
if __name__ == '__main__':
    pred = torch.rand((2,160,160, 9))
    target = torch.rand((2,160,160,8))
    loss_ = loss(pred, target)