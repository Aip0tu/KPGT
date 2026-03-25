from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, r2_score
import pandas as pd
import os
import numpy as np

try:
    import torch
except ImportError:
    torch = None

class Evaluator:
    """统一封装分类与回归任务的常用评估指标。"""
    def __init__(self, name, eval_metric, n_tasks, mean=None, std=None):
        self.name = name
        self.eval_metric = eval_metric
        self.n_tasks = n_tasks
        self.mean = mean
        self.std = std

    def _parse_and_check_input(self, y_true, y_pred, valid_ids=None):
        # 如果输入是张量，先移动到 CPU 并转成 numpy，便于统一计算指标。
        if torch is not None and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if torch is not None and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        # 检查输入类型与形状是否合法。
        if not isinstance(y_true, np.ndarray):
            raise RuntimeError('Arguments to Evaluator need to be either numpy ndarray or torch tensor')

        if not y_true.shape == y_pred.shape:
            raise RuntimeError('Shape of y_true and y_pred must be the same')

        if not y_true.ndim == 2:
            raise RuntimeError('y_true and y_pred mush to 2-dim arrray, {}-dim array given'.format(y_true.ndim))

        if not y_true.shape[1] == self.n_tasks:
            raise RuntimeError('Number of tasks for {} should be {} but {} given'.format(self.name, self.n_tasks, y_true.shape[1]))
        if valid_ids is not None and isinstance(valid_ids, torch.Tensor):
            valid_ids = valid_ids.detach().cpu().numpy()
        return y_true, y_pred, valid_ids


    def eval(self, y_true, y_pred, valid_ids=None):
        y_true, y_pred, valid_ids = self._parse_and_check_input(y_true, y_pred, valid_ids)
        if self.eval_metric == 'rocauc':
            return self._eval_rocauc(y_true, y_pred)
        elif self.eval_metric == 'rocauc_resp':
            return self._eval_rocauc_resp(y_true, y_pred, valid_ids)
        elif self.eval_metric == 'ap':
            return self._eval_ap(y_true, y_pred)
        elif self.eval_metric == 'ap_resp':
            return self._eval_ap_resp(y_true, y_pred)
        elif self.eval_metric == 'rmse':
            return self._eval_rmse(y_true, y_pred)
        elif self.eval_metric == 'acc':
            return self._eval_acc(y_true, y_pred)
        elif self.eval_metric == 'mae':
            return self._eval_mae(y_true, y_pred)
        elif self.eval_metric == 'r2':
            return self._eval_r2(y_true, y_pred)
        else:
            raise ValueError('Undefined eval metric %s ' % (self.eval_metric))

    def _eval_rocauc(self, y_true, y_pred):
        '''
            计算所有任务平均的 ROC-AUC
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            # AUC 需要同时存在正负样本。
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # 忽略缺失标签。
                is_labeled = y_true[:,i] == y_true[:,i]
                rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return sum(rocauc_list)/len(rocauc_list)

    def _eval_rocauc_resp(self, y_true, y_pred, valid_ids=None):
        '''
            逐任务计算 ROC-AUC，并返回列表
        '''

        rocauc_list = []

        for i in range(y_true.shape[1]):
            # AUC 需要同时存在正负样本。
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # 忽略缺失标签，并可选地限制到有效样本子集。
                is_labeled = y_true[:,i] == y_true[:,i]
                if valid_ids is not None:
                    is_labeled = np.logical_and(is_labeled, valid_ids[:, i])
                if len(y_true[is_labeled,i] != 0):
                    rocauc_list.append(roc_auc_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        if len(rocauc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        return rocauc_list


    def _eval_ap(self, y_true, y_pred):
        '''
            计算所有任务平均的 AP
        '''

        ap_list = []

        for i in range(y_true.shape[1]):
            # AP 同样要求同时存在正负样本。
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # 忽略缺失标签。
                is_labeled = y_true[:,i] == y_true[:,i]
                if len(y_true[is_labeled,i] != 0):
                    ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

                    ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        return sum(ap_list)/len(ap_list)
    def _eval_ap_resp(self, y_true, y_pred):
        '''
            逐任务计算 AP，并返回列表
        '''

        ap_list = []

        for i in range(y_true.shape[1]):
            # AP 同样要求同时存在正负样本。
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                # 忽略缺失标签。
                is_labeled = y_true[:,i] == y_true[:,i]
                ap = average_precision_score(y_true[is_labeled,i], y_pred[is_labeled,i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        return ap_list

    def _eval_rmse(self, y_true, y_pred):
        '''
            计算所有任务平均的 RMSE
        '''
        rmse_list = []
        for i in range(y_true.shape[1]):
            # 忽略缺失标签。
            is_labeled = y_true[:,i] == y_true[:,i]
            if (self.mean is not None) and (self.std is not None):
                rmse_list.append(np.sqrt(((y_true[is_labeled,i] - (y_pred[is_labeled,i]*self.std[i]+self.mean[i]))**2).mean()))
            else:
                rmse_list.append(np.sqrt(((y_true[is_labeled,i] - y_pred[is_labeled,i])**2).mean()))
        return sum(rmse_list)/len(rmse_list)
    
    def _eval_mae(self, y_true, y_pred):
        '''
            计算所有任务平均的 MAE
        '''
        mae_list = []

        for i in range(y_true.shape[1]):
            # 忽略缺失标签。
            is_labeled = y_true[:,i] == y_true[:,i]
            if (self.mean is not None) and (self.std is not None):
                mae_list.append(mean_absolute_error(y_true[:,i], y_pred[:,i]*self.std[i]+self.mean[i]))
            else:
                mae_list.append(mean_absolute_error(y_true[:,i], y_pred[:,i]))

        return sum(mae_list)/len(mae_list)
    def _eval_r2(self, y_true, y_pred):
        '''
            计算每个任务的 R2
        '''
        r2_list = []

        for i in range(y_true.shape[1]):
            # 忽略缺失标签。
            is_labeled = y_true[:,i] == y_true[:,i]
            if (self.mean is not None) and (self.std is not None):
                r2_list.append(r2_score(y_true[is_labeled,i], y_pred[is_labeled,i]*self.std[i]+self.mean[i]))
            else:
                r2_list.append(r2_score(y_true[is_labeled,i], y_pred[is_labeled,i]))

        return r2_list

    def _eval_acc(self, y_true, y_pred):
        acc_list = []

        for i in range(y_true.shape[1]):
            is_labeled = y_true[:,i] == y_true[:,i]
            correct = y_true[is_labeled,i] == y_pred[is_labeled,i]
            acc_list.append(float(np.sum(correct))/len(correct))

        return sum(acc_list)/len(acc_list)
