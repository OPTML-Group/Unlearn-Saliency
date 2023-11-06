import numpy as np
import torch
import torch.nn.functional as F


class black_box_benchmarks(object):
    def __init__(
        self,
        shadow_train_performance,
        shadow_test_performance,
        target_train_performance,
        target_test_performance,
        num_classes,
    ):
        """
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels.
        """
        self.num_classes = num_classes

        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance

        self.s_tr_corr = (
            np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels
        ).astype(int)
        self.s_te_corr = (
            np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels
        ).astype(int)
        self.t_tr_corr = (
            np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels
        ).astype(int)
        self.t_te_corr = (
            np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels
        ).astype(int)

        self.s_tr_conf = np.take_along_axis(
            self.s_tr_outputs, self.s_tr_labels[:, None], axis=1
        )
        self.s_te_conf = np.take_along_axis(
            self.s_te_outputs, self.s_te_labels[:, None], axis=1
        )
        self.t_tr_conf = np.take_along_axis(
            self.t_tr_outputs, self.t_tr_labels[:, None], axis=1
        )
        self.t_te_conf = np.take_along_axis(
            self.t_te_outputs, self.t_te_labels[:, None], axis=1
        )

        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

    def _log_value(self, probs, eps=1e-30):
        return -np.log(np.maximum(probs, eps))

    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[
            range(true_labels.size), true_labels
        ]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[
            range(true_labels.size), true_labels
        ]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)

    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
            te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre

    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        t_te_acc = 1 - np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + t_te_acc)
        print(
            "For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}".format(
                acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc
            )
        )
        return t_tr_acc, t_te_acc

    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(
                s_tr_values[self.s_tr_labels == num],
                s_te_values[self.s_te_labels == num],
            )
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
        t_tr_acc = t_tr_mem / (len(self.t_tr_labels) + 0.0)
        t_te_acc = t_te_non_mem / (len(self.t_te_labels) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + t_te_acc)
        print(
            "For membership inference attack via {n}, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}".format(
                n=v_name, acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc
            )
        )
        return t_tr_acc, t_te_acc

    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        ret = {}
        if (all_methods) or ("correctness" in benchmark_methods):
            ret["correctness"] = self._mem_inf_via_corr()
        if (all_methods) or ("confidence" in benchmark_methods):
            ret["confidence"] = self._mem_inf_thre(
                "confidence",
                self.s_tr_conf,
                self.s_te_conf,
                self.t_tr_conf,
                self.t_te_conf,
            )
        if (all_methods) or ("entropy" in benchmark_methods):
            ret["entropy"] = self._mem_inf_thre(
                "entropy",
                -self.s_tr_entr,
                -self.s_te_entr,
                -self.t_tr_entr,
                -self.t_te_entr,
            )
        if (all_methods) or ("modified entropy" in benchmark_methods):
            ret["m_entropy"] = self._mem_inf_thre(
                "modified entropy",
                -self.s_tr_m_entr,
                -self.s_te_m_entr,
                -self.t_tr_m_entr,
                -self.t_te_m_entr,
            )

        return ret


def collect_performance(data_loader, model, device):
    probs = []
    labels = []
    model.eval()

    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(data)
            prob = F.softmax(output, dim=-1)

        probs.append(prob)
        labels.append(target)

    return torch.cat(probs).cpu().numpy(), torch.cat(labels).cpu().numpy()


def MIA(
    retain_loader_train, retain_loader_test, forget_loader, test_loader, model, device
):
    shadow_train_performance = collect_performance(retain_loader_train, model, device)
    shadow_test_performance = collect_performance(test_loader, model, device)
    target_train_performance = collect_performance(retain_loader_test, model, device)
    target_test_performance = collect_performance(forget_loader, model, device)

    BBB = black_box_benchmarks(
        shadow_train_performance,
        shadow_test_performance,
        target_train_performance,
        target_test_performance,
        num_classes=10,
    )
    return BBB._mem_inf_benchmarks()
