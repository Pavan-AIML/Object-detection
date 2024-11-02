import torch
import numpy as np
from .compute_iou import compute_iou

def blockify_tensor(tensor: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack(
                [
                    tensor[..., block_size * j : block_size * (j + 1), block_size * i : block_size * (i + 1)]
                    for j in range(tensor.shape[-2] // block_size)
                ],
                dim=-3,
            )
            for i in range(tensor.shape[-1] // block_size)
        ],
        dim=-3,
    )
    
def get_obj_detections(model, data_loader, max_num_imgs=10):
    model.eval()
    device = list(model.parameters())[0].device
    results = {}
    for step, (imgs, targets) in enumerate(data_loader):
        if step>max_num_imgs:
            break
        print("idx: %d"%targets["image_id"].item())
        with torch.no_grad():
            preds = model(imgs) # list with batch_size num of elements
        indices = targets["image_id"].tolist() # list with batch_size num of elements
        for pred, idx in zip(preds, indices):
            results[idx] = {"boxes": pred["boxes"].cpu(), "scores": pred["scores"].cpu()}
    return results


def eval_obj_detect_fixIoU(dataset_test, results, IoU_th=0.5, conf_th=0.8, verbose=True):
    """Evaluates the detections based on one IoU threshold IoU_th
    """

    # Get the ground truth annotations for all images
    gt = {}
    for idx in results.keys():
        target = dataset_test._get_annotation(idx)
        # get visible bboxes
        bbox = target['boxes'][target['visibilities'].gt(dataset_test._vis_threshold)]
        gt[idx] = bbox.cpu()

    # Get the evaluation for all images
    P, P_pred, TP = 0, 0, 0
    for idx in results.keys():
        bbox_gt = gt[idx]
        bbox_pred_all= results[idx]['boxes'].cpu()
        confident_idxs = results[idx]['scores']>conf_th
        bbox_pred = bbox_pred_all[confident_idxs]
        # Loop through gt bboxes in image and mark TPs and FPs
        gt_idx_found = np.zeros(len(bbox_gt))
        pred_idx_found = np.zeros(len(bbox_pred))
        if len(bbox_gt) == 0 or len(bbox_pred) == 0:
            continue
        # Loop through all predictions and find the matching gt bbox
        for i_pred, box_pred in enumerate(bbox_pred):
            ovmax = -np.inf

            # Compute the intersection over union (IoU) for
            # a batch of gt bounding boxes and one predicition.
            # - store the maximum IoU in the variable ovmax.
            # - store the index of the detection with highest overlap in jmax.

            overlaps = compute_iou(bbox_gt, box_pred[None].expand_as(bbox_gt))
            ovmax, jmax_gt = torch.max(overlaps, dim=0)
            #print("found pred_box %d to overlap with gt_box %d with overlap: %0.1f"%(i_pred,jmax_gt.item(),ovmax.item()))

            if ovmax > IoU_th:
                # mark gt box as found
                gt_idx_found[jmax_gt] = 1.
                # mark box_pred as found
                pred_idx_found[i_pred] = 1.
        P += len(bbox_gt)
        P_pred += len(bbox_pred)
        TP += pred_idx_found.sum()
        #FP += (1.0-pred_idx_found).sum()
        #FN += (1.0-gt_idx_found).sum()
        if verbose:
            print(conf_th)
            print("idx: %d"%idx)
            print('\t %d\t P: num of gt boxes '%len(bbox_gt))
            print('\t %d\t P_pred: num of pred boxes '%len(bbox_pred))
            print('\t %d\t TP: num of gt boxes found'%(gt_idx_found.sum()))
            print('\t %d\t TP: num of pred boxes found '%(pred_idx_found.sum()))
            print('\t %d\t FP: num of pred boxes not found '%((1.0-pred_idx_found).sum()))
            print('\t %d\t FN: num of gt boxes not found '%((1.0-gt_idx_found).sum()))

    prec = TP/P_pred
    rec = TP/P

    return prec, rec
