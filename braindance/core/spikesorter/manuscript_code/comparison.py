import warnings

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import sys


class DummySorter:
    # Barebones sorter just for Comparison
    def __init__(self, spike_times, name="Sorter"):
        self.spike_times = spike_times
        self.name = name
        
    def get_spike_times(self):
        return self.spike_times

    def __len__(self):
        return len(self.spike_times)

    @staticmethod
    def units_to_sorter(units, name="Sorter"):
        # Convert an interable of units to DummySorter
        return DummySorter([unit.spike_train for unit in units], name=name)


class Comparison:
    """Class to compare two spike sorters"""
    def __init__(self, sorter1: DummySorter, sorter2: DummySorter, delta_time=0.4, match_score=0.5,
                 score_formula=1.5):
        """
        :param sorter1:
        :param sorter2:
        :param delta_time:
            If two spikes are within delta_time (in ms) of each other, they are considered the same spike
        :param match_score:
            If two units have an overlap score greater than match_score, they are considered matching.
        :param score_formula: 1.5, 1, or 2
            agreement_score between 2 units =
                1.5: num_matches / (num_spikes_sorter1 + num_spikes_sorter2 - num_matches)
                1: num_matches / (num_spikes_sorter1)
                2: num_matches / (num_spikes_sorter2)
        """

        self.sorter1 = sorter1
        self.sorter2 = sorter2

        self.score_formula = score_formula

        # Get spike times
        spike_times1 = sorter1.get_spike_times()
        spike_times2 = sorter2.get_spike_times()

        # Get match counts and agreement scores
        match_counts, agreement_scores = Comparison.get_match_count_and_agreement_scores(spike_times1, spike_times2, delta_time, score_formula)
        self.agreement_scores = agreement_scores

        # Make matches
        self.match12, self.match21 = Comparison.make_hungarian_match(agreement_scores, match_score)
        # self.match12, self.match21 = Comparison.make_best_match(agreement_scores, match_score)

    # For manuscript plotting
    # def summary(self, SVG_PATH):
    
    def summary(self):
        sorter1 = self.sorter1
        sorter2 = self.sorter2
        hungarian_match_21 = self.match21
        agreement_scores = self.agreement_scores
        
        print(f"Num {sorter1.name}: {len(sorter1)}")
        print(f"Num {sorter2.name}: {len(sorter2)}")
        print()
        
        if self.score_formula == 1.5:
            print("Spikeinterface formula")
        elif self.score_formula == 1:
            print(f"#matches/#rt_sort")
        elif self.score_formula == 2:
            # print(f"#matches/#kilosort")
            print(f"#matches/#other_sorter")
        print(f"{sum(hungarian_match_21 != -1)}/{len(sorter1)} matches")
        print(f"{np.sum(np.max(agreement_scores, axis=1) >= 0.5)}/{len(sorter1)} max matches")
        
        # print(f"\nFor {sorter1.name}:")
        # max_matches_bool = np.max(agreement_scores, axis=1) >= 0.5
        # print(f"Num max matches: {np.sum(max_matches_bool)}")
        # max_matches_ind = np.unique(np.argmax(agreement_scores[max_matches_bool, :], axis=1))
        # print(f"Num unique max matches: {max_matches_ind.size}")

        # print(f"\nFor {sorter2.name}:")
        # max_matches_bool = np.max(agreement_scores, axis=0) >= 0.5
        # print(f"Num max matches: {np.sum(max_matches_bool)}")
        # max_matches_ind = np.unique(np.argmax(agreement_scores[:, max_matches_bool], axis=0))
        # print(f"Num unique max matches: {max_matches_ind.size}")

        # Plot histogram of agreement scores
        # plt.title("All agreement scores.")
        # plot_hist_percents(agreement_scores.flatten(), range=(0, 1), bins=20)
        # plt.xlabel("Agreement score")
        # plt.xticks([x/10 for x in range(11)])
        # plt.xlim(0, 1)
        # plt.show()
        
        # plt.title(f"For {sorter1.name}'s units, agreement scores.")
        plt.title(f"Overlap scores for {sorter1.name} units")
        # plot_hist_percents(np.max(agreement_scores, axis=1), range=(0, 1), bins=20)
        plt.hist(np.max(agreement_scores, axis=1), range=(0, 1), bins=20)
        plt.xlabel("Overlap score")
        plt.xticks([x/10 for x in range(11)])
        plt.ylabel("#units")
        plt.xlim(0, 1)
        # plt.ylim(0, 70)
        plt.show()

        # plt.title(f"For {sorter2.name}'s units, agreement scores.")
        plt.title(f"Overlap scores for {sorter2.name} units")
        # plot_hist_percents(np.max(agreement_scores, axis=0), range=(0, 1), bins=20)
        plt.hist(np.max(agreement_scores, axis=0), range=(0, 1), bins=20)
        plt.xlabel("Overlap score")
        plt.xticks([x / 10 for x in range(11)])
        plt.xlim(0, 1)
        plt.ylabel("#units")
        # plt.ylim(0, 70)
        plt.show()
        
        # # region Format for Nature manuscript
        # # Any of these can be None to not use these defined values
        # XTICKS = [0, 0.5, 1]
        # YTICKS = [0, 100, 200, 300]
        # YLIM = [0, 360]
        # """
        # YTICKS that best fit each sorter's three histograms (three different score formulas)
        # herdingspikes: [0, 50, 100]
        # kilosort2: [0, 125, 250]
        # ironclust: [0, 35, 70]
        # tridesclous: [0, 25, 50]
        # spykingcircus: [0, 180, 360]
        # hdsort: [0, 75, 150]
        # """
                
        # from pathlib import Path
        # # SAVE_ROOT = Path(f"/data/MEAprojects/dandi/000034/sub-mouse412804/rt_sort/240319/overlap_score_histograms/herdingspikes")  
        # # SVG_PATH = SAVE_ROOT / "spikeinterface_formula.svg"
        # # SVG_PATH = SAVE_ROOT / "num_matches_over_num_rt_sort.svg"
        # # SVG_PATH = SAVE_ROOT / "num_matches_over_num_other_sorter.svg"
        # # SVG_PATH = SAVE_ROOT / "scatter.svg"
        
        # TICKS_PATH = SVG_PATH.parent / f"{SVG_PATH.name.split('.')[0]}_ticks.txt"  # Where to save ticks
        
        # SVG_PATH.parent.mkdir(exist_ok=True, parents=True)
        
        # ax = plt.gca()
        # # Hide top and right spines
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # # Increase thickness of the bottom and left spines
        # ax.spines["bottom"].set_linewidth(2.5)
        # ax.spines["left"].set_linewidth(2.5)

        # # Increase thickness of tick marks
        # ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')

        # # Hide labels
        # ax.set_title("")
        # ax.set_xlabel("")
        # ax.set_ylabel("")

        # # Update ticks
        # if XTICKS is not None:
        #     ax.set_xticks(XTICKS, [''] * len(XTICKS))
        #     ax.set_xlim(XTICKS[0], XTICKS[-1])
        # if YTICKS is not None:
        #     ax.set_yticks(YTICKS, [''] * len(YTICKS))
        #     if YLIM is None:
        #         ax.set_ylim(YTICKS[0], YTICKS[-1])
        #     else:
        #         ax.set_ylim(*YLIM)

        # # Save figure
        # plt.savefig(SVG_PATH, format="svg")
        # plt.close()
        
        # # Save ticks
        # with open(TICKS_PATH, "w") as file:
        #     file.write(f"x-ticks: {XTICKS}\ny-ticks: {YTICKS}")
        # # endregion
        
        # plt.show()
        
        # # Get precision and recall for spike_times1. Assumes spikes from spike_times2 is ground truth
        # matches = hungarian_match_12  # Choose hungarian match or best_match
        # precisions = {}
        # recalls = {}

        # # The ith element in spike_times_matching_1 is the spike_train that matches with the ith element in spike_times_matching2
        # self.spike_times_matching1 = []  # The spike times from spike_times1
        # self.spike_times_matching2 = []  # The spike times from spike_times2
        # self.matching_uids = []  # (uid of sorter1, uid of sorter2)

        # for i in range(len(matches)):
        #     match = matches[i]
        #     if match != -1:  # If -1, has no match
        #         num_matches = match_counts[i, match]
        #         precisions[i] = num_matches / len(spike_times1[i])
        #         recalls[i] = num_matches / len(spike_times2[match])

        #         self.spike_times_matching1.append(spike_times1[i])
        #         self.spike_times_matching2.append(spike_times2[match])
        #         self.matching_uids.append((i, match))

        # # Plot precisions and recalls
        # PRECISION_COLOR = "blue"
        # RECALL_COLOR = "green"
        # for i in precisions:
        #     precision = precisions[i]
        #     recall = recalls[i]
        #     plt.scatter(0, precision, color=PRECISION_COLOR, alpha=0.6)
        #     plt.scatter(1, recall, color=RECALL_COLOR, alpha=0.6)
        # plt.title(f"{sorter1.name}'s units' performances if {sorter2.name} is ground truth.")
        # plt.xticks(ticks=[0, 1], labels=["Precision", "Recall"])
        # plt.ylim(match_score, 1)
        # plt.show()

    def plot_spike_map(self,
                       size_scale=300, spike_alpha=0.5,
                       color_sorter1="red", color_sorter2="blue", elec_color="black",
                       xlim=(0, 3850), ylim=(0, 2100),
                       xlabel="x (µm)", ylabel="y (µm)",
                       figsize=None):
        """
        Plot a map of sorter1 and sorter2 spikes
        Map will contain the location of all electrodes in recording.
        For each unit, a circle will be plotted at its location. The size will be based on the relative
        number of the unit's spikes. The two sorters' spikes will have different colors

        :param recording:
            Path to recording or Recording object
        :param size_scale:
            Size of dot = num_spikes / max_num_spikes * size_scale
        :param spike_alpha
            Alpha of dot
        :param color_sorter1:
        :param color_sorter2:
        :param elec_color:
        :param xlim
        :param ylim
        :param xlabel:
            x-label of plot
        :param ylabel:
            y-label of plot
            y (µm)
        """

        channel_locs = self.sorter1.recording.get_channel_locations()

        if figsize is not None:
            plt.figure(figsize=figsize)

        plt.scatter(channel_locs[:, 0], channel_locs[:, 1], s=1, c=elec_color, marker=",", zorder=-100)

        locs = []
        nums_spikes = []
        for sorter in (self.sorter2, self.sorter1):
            # for spike_times in sorter.spike_times:
            for unit in sorter:
                locs.append(channel_locs[unit.chan])
                nums_spikes.append(len(unit))

        locs = np.array(locs)
        nums_spikes = np.array(nums_spikes)
        colors = [color_sorter2] * len(self.sorter2) + [color_sorter1] * len(self.sorter1)
        plt.scatter(locs[:, 0], locs[:, 1], s=nums_spikes / np.max(nums_spikes) * size_scale, c=colors, alpha=spike_alpha)

        plt.title("Spike map.")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        xlim = xlim if xlim is not None else (0, np.max(channel_locs[:, 0]))
        ylim = ylim if ylim is not None else (0, np.max(channel_locs[:, 1]))
        plt.xlim(xlim)
        plt.ylim(ylim)

        plt.scatter(-1000, -1000, s=np.mean(nums_spikes[len(self.sorter2):]) / np.max(nums_spikes) * size_scale, c=color_sorter1, alpha=spike_alpha, label=self.sorter1.name)
        plt.scatter(-1000, -1000, s=np.mean(nums_spikes[:len(self.sorter2)]) / np.max(nums_spikes) * size_scale, c=color_sorter2, alpha=spike_alpha, label=self.sorter2.name)
        # plt.legend(loc="best")

        plt.tight_layout()
        plt.show()

    def get_max_agreement_scores(self):
        # Get maximum agreement score for each unit
        # (for sorter1 units, for sorter2 units)
        return self.agreement_scores.max(axis=1), self.agreement_scores.max(axis=0)

    # def get_hungarian_agreement_scores(self, sorter=1):
    #     """
    #     Agreement scores for sorter's units when at most one unit can be matched with one other unit
    #     """
        
    #     if sorter not in {1, 2}:
    #         raise ValueError('param "sorter" must be 1 or 2')

    #     if sorter == 1:
    #         agreement_scores = self.agreement_scores
    #     else:
    #         agreement_scores = self.agreement_scores.T
    #     hungarian_scores = []
        
    #     scores = agreement_scores.copy()
    #     # scores[scores < min_score] = 0

    #     [inds1, inds2] = linear_sum_assignment(-scores)

    #     hungarian_match_12 = np.full((scores.shape[0],), -1)
    #     hungarian_match_21 = np.full((scores.shape[1],), -1)

    #     for i1, i2 in zip(inds1, inds2):
    #         if agreement_scores[i1, i2] >= min_score:
    #             hungarian_match_12[i1] = i2
    #             hungarian_match_21[i2] = i1

    #     return hungarian_match_12, hungarian_match_21


    def plot_line_comps(self):
        fig, axes = plt.subplots(2, 2, tight_layout=True, figsize=(7, 7))
        self.plot_line_comp_all(1, axes[0][0])
        self.plot_line_comp_max(1, axes[1][0])
        self.plot_line_comp_all(2, axes[0][1])
        self.plot_line_comp_max(2, axes[1][1])
        plt.show()

    def plot_line_comp_all(self, sorter=1, axis=None):
        """
        Plot line comparison plot for all units

        :param sorter:
            Each line will be a unit from sorter (1 or 2)
            x-axis will be units from the other sorter
        """
        if sorter not in {1, 2}:
            raise ValueError('param "sorter" must be 1 or 2')

        if axis is None:
            fig, axis = plt.subplots(1, 1)
            show = True
        else:
            show = False


        if sorter == 1:
            agreement_scores = self.agreement_scores
            sorter_lines = self.sorter1
            sorter_x = self.sorter2
        else:
            agreement_scores = self.agreement_scores.T
            sorter_lines = self.sorter2
            sorter_x = self.sorter1

        agreement_scores = -np.sort(-agreement_scores, axis=1)
        for i in range(agreement_scores.shape[0]):
            scores = agreement_scores[i, :]
            axis.plot(range(scores.size), scores, color="black", alpha=0.1)

        axis.set_ylim(0, 1)
        axis.set_title(f"Each line is a {sorter_lines.name} unit")
        axis.set_ylabel("Agreement score")
        axis.set_xlabel(f"Descending sorted index of {sorter_x.name} unit")

        X_MAX = 14
        axis.set_xlim(0, 14)  # agreement_scores.shape[0])
        axis.set_xticks(range(X_MAX+1))

        if show:
            plt.show()

    def plot_line_comp_max(self, sorter=1, axis=None):
        """
        Plot line comparison plot
            1. Each line is a unit from :param sorter:
            2. Each unit from the other sorter is only paired with
               one unit from :param sorter: (max match)
        """
        if sorter not in {1, 2}:
            raise ValueError('param "sorter" must be 1 or 2')

        if axis is None:
            fig, axis = plt.subplots()
            show = True
        else:
            show = False

        if sorter == 1:
            agreement_scores = self.agreement_scores
            sorter_lines = self.sorter1
            sorter_x = self.sorter2
        else:
            agreement_scores = self.agreement_scores.T
            sorter_lines = self.sorter2
            sorter_x = self.sorter1

        # agreement_scores has shape (num_lines, num_units_in_lines)
        lines = [[] for _ in range(agreement_scores.shape[0])]
        matches = [[] for _ in range(agreement_scores.shape[0])]
        for j in range(agreement_scores.shape[1]):
            unit = agreement_scores[:, j]
            match = np.argmax(unit)
            lines[match].append(unit[match])
            matches[match].append(j)

        x_values = range(agreement_scores.shape[1])
        for line in lines:
            add_y = len(x_values) - len(line)
            axis.plot(x_values, sorted(line, reverse=True) + [0]*add_y, color="black", alpha=0.1)

        axis.set_ylim(0, 1)
        axis.set_title(f"Each line is a {sorter_lines.name} unit")
        axis.set_ylabel("Agreement score")
        axis.set_xlabel(f"Descending sorted index of {sorter_x.name} unit")

        X_MAX = 14
        axis.set_xlim(0, 14)  # agreement_scores.shape[0])
        axis.set_xticks(range(X_MAX+1))

        if show:
            plt.show()

        return matches

    @staticmethod
    def get_match_count_and_agreement_scores(spike_times1, spike_times2, delta_time=0.4, score_formula=1.5):
        """
        Get agreement scores of spike times

        From SpikeInterface paper:
            score = num_matches / (num_spikes_in_train1 + num_spikes_in_train2 - num_matches)

        :param spike_times1:
        :param spike_times2:
        :param delta_time:
        :param score_formula: 1.5, 1, or 2
            agreement_score =
                1.5: num_matches / (num_spikes_sorter1 + num_spikes_sorter2 - num_matches)
                1: num_matches / (num_spikes_sorter1)
                2: num_matches / (num_spikes_sorter2)

        :return:
            np.array with shape (len(spike_times1), len(spike_times2)) showing the match scores between the units
        :return:
            np.array with shape (len(spike_times1), len(spike_times2)) showing the agreement scores between the units
        """

        n = len(spike_times1)
        m = len(spike_times2)

        # warnings.warn("Using Comparison.count_matching_events_si which is inaccurate when many spikes in succession match between the two sorters", UserWarning)

        match_counts = np.zeros((n, m), dtype=int)
        agreement_scores = np.zeros((n, m), dtype=float)
        for i in range(n):
            train1 = spike_times1[i]
            for j in range(m):
                train2 = spike_times2[j]
                matches = Comparison.count_matching_events_si(train1, train2, delta_time)
                match_counts[i, j] = matches
                try: 
                    if score_formula == 1.5:
                        agreement_scores[i, j] = matches / (len(train1) + len(train2) - matches)
                    elif score_formula == 2:
                        agreement_scores[i, j] = matches / len(train2)
                    elif score_formula == 1:
                        agreement_scores[i, j] = matches / len(train1)
                    else:
                        raise ValueError(f"argument 'score_formula' must be 1.5, 1, or 2, not '{score_formula}'")
                except ZeroDivisionError:
                    agreement_scores[i, j] = 0
        return match_counts, agreement_scores

    # @staticmethod
    def count_matching_events_si(times1, times2, delta: float = 0.4, return_matches=False):
        """
        From spikeinterface

        Counts matching events.

        Parameters
        ----------
        times1: list
            List of spike train 1 (in ms)
        times2: list
            List of spike train 2 (in ms)
        delta: int
            ms for considering matching events

        Returns
        -------
        matching_count: int
            Number of matching events

        Does not work for
        times1 = [1, 1.8]
        times2 = [1.4, 2.2]
        i.e. one spike in times1 can belong to multiple in times2

        """

        times_concat = np.concatenate((times1, times2))
        membership = np.concatenate((np.ones(len(times1)) * 1, np.ones(len(times2)) * 2))
        indices = times_concat.argsort()
        times_concat_sorted = times_concat[indices]
        membership_sorted = membership[indices]
        # diffs = times_concat_sorted[1:] - times_concat_sorted[:-1]
        # inds = np.where((diffs <= delta) & (membership_sorted[:-1] != membership_sorted[1:]))[0]
        diffs = times_concat_sorted[1:] - times_concat_sorted[:-1] - delta
        inds = np.where((diffs <= sys.float_info.epsilon) & (membership_sorted[:-1] != membership_sorted[1:]))[0]
        if not return_matches:
            if len(inds) == 0:
                return 0
            inds2 = np.where(inds[:-1] + 1 != inds[1:])[0]  # Prevents a spike being matched with more than one other spike
            return len(inds2) + 1
        else:
            return inds
    
    @staticmethod
    def count_matching_events(times1, times2, delta=0.4):
        count = 0
        ptr1 = ptr2 = 0
        while ptr1 < len(times1) and ptr2 < len(times2):
            diff = abs(times1[ptr1] - times2[ptr2]) - delta
            if diff <= 1e-4:
                count += 1
                ptr1 += 1
                ptr2 += 1
            elif times1[ptr1] < times2[ptr2]:
                ptr1 += 1
            else:
                ptr2 += 1
        
        return count
        
    @staticmethod
    def get_matching_events(times1, times2, delta=0.4):
        count = 0
        ptr1, ptr2 = 0, 0
        matching_times1 = []
        unmatching_times1 = []
        matching_times2 = []
        unmatching_times2 = []
        while ptr1 < len(times1) and ptr2 < len(times2):            
            diff = abs(times1[ptr1] - times2[ptr2]) - delta
            if diff <= 1e-4:
                matching_times1.append(times1[ptr1])
                matching_times2.append(times2[ptr2])
                count += 1
                ptr1 += 1
                ptr2 += 1
            elif times1[ptr1] < times2[ptr2]:
                unmatching_times1.append(times1[ptr1])
                ptr1 += 1
            else:
                unmatching_times2.append(times2[ptr2])
                ptr2 += 1
        
        if ptr1 < len(times1):
            unmatching_times1.extend(times1[ptr1:])
        else:
            unmatching_times2.extend(times2[ptr2:])

        return matching_times1, unmatching_times1, unmatching_times2

    @staticmethod
    def make_hungarian_match(agreement_scores: np.ndarray, min_score: float):
        """
        From SpikeInterface

        Given an agreement matrix and a min_score threshold for a match
        return the "optimal" match with the "hungarian" algo.
        This use internally the scipy.optimze.linear_sum_assignment implementation.

        """

        # Threshold the matrix
        scores = agreement_scores.copy()
        scores[scores < min_score] = 0

        [inds1, inds2] = linear_sum_assignment(-scores)

        hungarian_match_12 = np.full((scores.shape[0],), -1)
        hungarian_match_21 = np.full((scores.shape[1],), -1)

        for i1, i2 in zip(inds1, inds2):
            if agreement_scores[i1, i2] >= min_score:
                hungarian_match_12[i1] = i2
                hungarian_match_21[i2] = i1

        return hungarian_match_12, hungarian_match_21

    @staticmethod
    def make_best_match(agreement_scores: np.ndarray, min_score: float):
        """
        From spikeinterface.comparison.make_best_match

        Given an agreement matrix and a min_score threshold.
        return a dict a best match for each units independently of others.
        """
        scores = agreement_scores.copy()

        best_match_12 = np.full((scores.shape[0],), -1)
        best_match_12[:] = -1
        for i1 in range(len(best_match_12)):
            ind_max = np.argmax(scores[i1, :])
            if scores[i1, ind_max] >= min_score:
                best_match_12[i1] = ind_max

        best_match_21 = np.full((scores.shape[1],), -1)
        best_match_21[:] = -1
        for i2 in range(len(best_match_21)):
            ind_max = np.argmax(scores[:, i2])
            if scores[ind_max, i2] >= min_score:
                best_match_21[i2] = ind_max

        return best_match_12, best_match_21

    @staticmethod
    def get_matching_spikes(times1, times2, delta=0.4):
        assert NotImplementedError, "This doesn't work properly, because of something like spikes are double counted or missed"
        
        matched = []
        unmatched1 = []

        already_matched = set()
        for st1 in times1:
            idx = np.searchsorted(times2, st1)
            idx_left = idx - 1
            while idx_left in already_matched:
                idx_left -= 1
            if idx_left >= 0:
                left = times2[idx_left]
            else:
                left = -np.inf

            idx_right = idx
            while idx_right in already_matched:
                idx_right += 1
            if idx_right < len(times2):
                right = times2[idx_right]
            else:
                right = np.inf

            if right - st1 < st1 - left:
                if right - st1 <= delta:
                    matched.append(st1)
                    already_matched.add(idx_right)
                else:
                    unmatched1.append(st1)
            else:
                if st1 - left <= delta:
                    matched.append(st1)
                    already_matched.add(idx_left)
                else:
                    unmatched1.append(st1)

        unmatched2 = [times2[i] for i in range(len(times2)) if i not in already_matched]

        return matched, unmatched1, unmatched2

    @staticmethod
    def full(sorter1, sorter2):
        comp_1_5 = Comparison(sorter1, sorter2, score_formula=1.5)
        comp_1_5.summary()

        comp_1 = Comparison(sorter1, sorter2, score_formula=1)
        comp_1.summary()

        comp_2 = Comparison(sorter1, sorter2, score_formula=2)
        comp_2.summary()
        comp_2.plot_line_comps()
        
        return comp_1_5, comp_1, comp_2

    @staticmethod
    def count_same_frames(times1, times2):
        # Count number of times the same time (frame perfect) occurs in times1 and times2
        return len(set(times1).intersection(times2))

    @staticmethod
    def count_all_overlaps(times1, times2, delta=0.4):
        """
        Used for finding number of overlaps when spike trains across all units are concatenated
        
        Params:
        times1 and times2
            Need to be a list of lists (each list is the spike times of a unit)
        """
        all_times1 = []
        for t in times1:
            all_times1.extend(t)
        all_times1 = np.unique(all_times1)
        
        all_times2 = []
        for t in times2:
            all_times2.extend(t)
        all_times2 = np.unique(all_times2)
        
        matches = Comparison.count_matching_events(all_times1, all_times2, delta=delta)
        print(matches/len(all_times1))
        print(matches/len(all_times2))
    
    @staticmethod
    def plot_comp_1_2_scatter(comp_1, comp_2, SVG_PATH):
        # Plot scatter plot for each sorter, for each unit, showing matches/sorter1 vs. matches/sorter2 
        for sorter_idx in (1,): # (0, 1):
            all_scores_1 = np.max(comp_1.agreement_scores, axis=(sorter_idx+1)%2) 
            all_scores_2 = np.max(comp_2.agreement_scores, axis=(sorter_idx+1)%2)
            for score_1, score_2 in zip(all_scores_1, all_scores_2):
                plt.scatter(score_1, score_2, color="black", alpha=0.6)
            plt.title(f"{comp_1.sorter1.name if sorter_idx == 0 else comp_1.sorter2.name} units")
            plt.xlabel(f"#matches/{comp_1.sorter1.name}")
            plt.ylabel(f"#matches/{comp_2.sorter2.name}")
            plt.xlim(-0.01, 1.01)
            plt.ylim(-0.01, 1.01)
            plt.gca().set_aspect("equal")
            # plt.show()
            
            # region Format for Nature manuscript
            # Any of these can be None to not use these defined values
            XTICKS = [0, 0.5, 1]
            YTICKS = [0, 0.5, 1]
            YLIM = None
                    
            from pathlib import Path
            # SAVE_ROOT = Path(f"/data/MEAprojects/dandi/000034/sub-mouse412804/rt_sort/240319/overlap_score_histograms/herdingspikes")  
            # SVG_PATH = SAVE_ROOT / "spikeinterface_formula.svg"
            # SVG_PATH = SAVE_ROOT / "num_matches_over_num_rt_sort.svg"
            # SVG_PATH = SAVE_ROOT / "num_matches_over_num_other_sorter.svg"
            # SVG_PATH = SAVE_ROOT / "scatter.svg"
            
            TICKS_PATH = SVG_PATH.parent / f"{SVG_PATH.name.split('.')[0]}_ticks.txt"  # Where to save ticks
            
            SVG_PATH.parent.mkdir(exist_ok=True, parents=True)
            
            ax = plt.gca()
            # Hide top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Increase thickness of the bottom and left spines
            ax.spines["bottom"].set_linewidth(2.5)
            ax.spines["left"].set_linewidth(2.5)

            # Increase thickness of tick marks
            ax.tick_params(axis='both', direction='out', length=6, width=2.5, colors='black')

            # Hide labels
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Update ticks
            if XTICKS is not None:
                ax.set_xticks(XTICKS, [''] * len(XTICKS))
                ax.set_xlim(XTICKS[0], XTICKS[-1])
            if YTICKS is not None:
                ax.set_yticks(YTICKS, [''] * len(XTICKS))
                if YLIM is None:
                    ax.set_ylim(YTICKS[0], YTICKS[-1])
                else:
                    ax.set_ylim(*YLIM)

            # Save figure
            plt.savefig(SVG_PATH, format="svg")
            plt.close()
            
            # Save ticks
            with open(TICKS_PATH, "w") as file:
                file.write(f"x-ticks: {XTICKS}\ny-ticks: {YTICKS}")
            # endregion
            
            plt.show()
    
"""
comp_1_5 = Comparison(prop_signal, kilosort, score_formula=1.5, delta_time=0.4)
comp_1_5.summary()
comp_1_5.plot_line_comps()

comp_1 = Comparison(prop_signal, kilosort, score_formula=1, delta_time=0.4)
comp_1.summary()
comp_1.plot_line_comps()

comp_2 = Comparison(prop_signal, kilosort, score_formula=2, delta_time=0.4)
comp_2.summary()
comp_2.plot_line_comps()
"""
