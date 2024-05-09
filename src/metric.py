import numpy as np

__all__ = ["Ct_index"]

def Ct_index(y_true: np.ndarray, y_score: np.ndarray, num_events, max_time, average: bool = False):
    def get_ct_index_mask(labels):
        batch_dim = labels.shape[0]

        tmp1 = np.repeat(np.expand_dims(labels[:, 1], axis=1), batch_dim, axis=1)
        tmp1 = tmp1 < tmp1.transpose(1, 0)

        tmp2 = np.repeat(np.expand_dims(labels[:, 0], axis=1), batch_dim, axis=1)

        tmp2 = tmp2 != 0
        tmp2[labels[:, 0] == 0, :] = 0

        mask = (tmp1 * tmp2).astype(int)

        return mask

    CIF = y_score.reshape( (-1, num_events, max_time) )
    batch_dim = CIF.shape[0]

    sample_idx = np.arange(batch_dim)
    k_event_idx = np.clip(y_true[:, 0] - 1, a_min=0, a_max=None)
    k_time_idx = y_true[:, 1]

    tmp1 = CIF[sample_idx, k_event_idx, k_time_idx]
    tmp1 = np.repeat(np.expand_dims(tmp1, axis=1), batch_dim, axis=1)

    CIF_ref = np.swapaxes(CIF, 2, 0)
    tmp2 = CIF_ref[k_time_idx, k_event_idx]

    tmp = (tmp1 > tmp2).astype(int)

    mask = get_ct_index_mask(y_true)

    tmp_num = np.repeat( np.sum(mask * tmp, axis=1, keepdims=True), num_events, axis=1)
    tmp_den = np.repeat( np.sum(mask, axis=1, keepdims=True), num_events, axis=1)

    k_masks = np.vstack( [y_true[:, 0] == k for k in range(1, num_events+1)] ).transpose(1, 0)
    tmp_num = k_masks * tmp_num
    tmp_den = k_masks * tmp_den

    tmp_num = np.sum(tmp_num, axis=0)
    tmp_den = np.sum(tmp_den, axis=0)

    ct_index = tmp_num / tmp_den

    if average:
        return np.sum(ct_index)
    return ct_index


if __name__ == "__main__":
    pass
