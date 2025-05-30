import numpy as np
import pandas as pd
import pytest
import curve_curator.thresholding as thresholding

#
# Helper functions
#
def get_empty_df(n_rows, decoy=False):
    # Create defaults
    df = pd.DataFrame({
        'Name' : range(n_rows),
        'Score' : np.full(n_rows, fill_value=0.0),
        'Decoy' : np.full(n_rows, fill_value=decoy),
    })

    # Make nice names
    name_label = 'Decoy' if decoy else 'Target'
    df['Name'] = name_label + ' ' + df['Name'].astype(str)
    return df


#
# Actual test classes
#

class TestS0:
    fc_lim = 1.0
    alpha = 0.05
    dfn = 5
    dfd = 10
    loc = 0
    scale = 1
    two_sided = True

    def test_fc_lims(self):
        expected_s0 = 0.4858672572170374
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

        s0 = thresholding.get_s0(-self.fc_lim, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

        expected_ratio = 2.0
        ratio = thresholding.get_s0(2.0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided) / \
                thresholding.get_s0(1.0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        np.isclose(expected_ratio, ratio)

    def test_alpha(self):
        expected_s0 = 1.0358756330851262
        s0 = thresholding.get_s0(self.fc_lim, 1.0, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

        expected_s0 = 0.38145809519312646
        s0 = thresholding.get_s0(self.fc_lim, 0.01, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

        expected_s0 = 0.0
        s0 = thresholding.get_s0(self.fc_lim, 0.0, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

        with pytest.raises(ValueError):
            s0 = thresholding.get_s0(self.fc_lim, 3.0, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
            s0 = thresholding.get_s0(self.fc_lim, -3.0, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)

    def test_dofs(self):
        expected_s0 = 0.0
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, 0.0, self.dfd, self.loc, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

        expected_s0 = 0.5186995593741331
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, 10.0, self.dfd, self.loc, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

        expected_s0 = 0.0
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, self.dfn, 0.0, self.loc, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

        expected_s0 = 0.6090252621538038
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, self.dfn, 100.0, self.loc, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

    def test_loc(self):
        expected_s0 = 0.4858672572170374
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, self.dfn, self.dfd, 0, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

        expected_s0 = 0.43701528619494895
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, self.dfn, self.dfd, 1, self.scale, self.two_sided)
        np.isclose(expected_s0, s0)

    def test_scale(self):
        expected_s0 = 0.6871200646693514
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, self.dfn, self.dfd, self.loc, 0.5, self.two_sided)
        np.isclose(expected_s0, s0)

        expected_s0 = 0.3435600323346757
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, self.dfn, self.dfd, self.loc, 2.0, self.two_sided)
        np.isclose(expected_s0, s0)

    def test_two_sided(self):
        expected_s0 = 0.5483396884699329
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, self.dfn, self.dfd, self.loc, self.scale, False)
        np.isclose(expected_s0, s0)

        expected_s0 = 0.4858672572170374
        s0 = thresholding.get_s0(self.fc_lim, self.alpha, self.dfn, self.dfd, self.loc, self.scale, True)
        np.isclose(expected_s0, s0)


class TestFCLim:
    s0 = 0.5
    alpha = 0.05
    dfn = 5
    dfd = 11
    loc = 0
    scale = 1
    two_sided = True

    def test_output_type(self):
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        assert type(fc) is tuple

    def test_output_shape(self):
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        assert len(fc) == 2

    def test_symmetric_output(self):
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        assert fc[0] == -fc[1]
        assert -fc[0] == fc[1]

    def test_fc_lims(self):
        expected_fc = 1.0054847365908508
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

        fc = thresholding.get_fclim(-self.s0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

        expected_ratio = 2.0
        ratio = thresholding.get_fclim(2.0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)[0] / \
                thresholding.get_fclim(1.0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)[0]
        np.isclose(expected_ratio, ratio)

    def test_alpha(self):
        expected_fc = 0.4811651050254145
        fc = thresholding.get_fclim(self.s0, 1.0, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

        expected_fc = 1.267058155831331
        fc = thresholding.get_fclim(self.s0, 0.01, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

        expected_fc = np.inf
        fc = thresholding.get_fclim(self.s0, 0.0, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

        with pytest.raises(ValueError):
            fc = thresholding.get_fclim(self.s0, 3.0, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)
            fc = thresholding.get_fclim(self.s0, -3.0, self.dfn, self.dfd, self.loc, self.scale, self.two_sided)

    def test_dofs(self):
        fc = thresholding.get_fclim(self.s0, self.alpha, 0.0, self.dfd, self.loc, self.scale, self.two_sided)
        assert all(np.isnan(fc))  # is nan output what we want ? or rather a raise error ?

        expected_fc = 0.9388386064558703
        fc = thresholding.get_fclim(self.s0, self.alpha, 10.0, self.dfd, self.loc, self.scale, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, 0.0, self.loc, self.scale, self.two_sided)
        assert all(np.isnan(fc))   # is nan output what we want  ? or rather a raise error ?

        expected_fc = 0.8209840068567296
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, 100.0, self.loc, self.scale, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

    def test_loc(self):
        expected_fc = 1.0054847365908508
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, 0, self.scale, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

        expected_fc = 1.1229423651804986
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, 1, self.scale, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

        expected_fc = 0.8723528847417039
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, -1, self.scale, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

    def test_scale(self):
        expected_fc = 0.7109850756229601
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, self.loc, 0.5, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

        expected_fc = 1.4219701512459202
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, self.loc, 2.0, self.two_sided)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

    def test_two_sided(self):
        expected_fc = 0.8949684718929517
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, False)
        assert all(np.isclose((-expected_fc, expected_fc), fc))

        expected_fc = 1.0054847365908508
        fc = thresholding.get_fclim(self.s0, self.alpha, self.dfn, self.dfd, self.loc, self.scale, True)
        assert all(np.isclose((-expected_fc, expected_fc), fc))


class TestFCtoPValueMap:
    fc_values = pd.Series([-np.inf, -2, -1, 0, 1, 2, np.inf])
    fc_lim = 0.5
    alpha = 0.1
    dfn = 5
    dfd = 11
    loc = 0.0
    scale = 1.0
    two_sided = True
    s0 = thresholding.get_s0(fc_lim, alpha, dfn, dfd, loc, scale, two_sided)
    p_values = thresholding.map_fc_to_pvalue_cutoff(fc_values, alpha, s0, dfn, dfd, loc, scale, two_sided).values

    def test_symmetry(self):
        assert all(self.p_values == self.p_values[::-1])

    def test_alpha_lim(self):
        log_p = -np.log10(self.alpha)
        assert np.isclose(self.p_values[0], log_p)
        assert np.isclose(self.p_values[6], log_p)

    def test_zero(self):
        assert self.p_values[3] == np.inf

    def test_all_values(self):
        expected_p = [1., 1.80652855, 3.25318495, np.inf, 3.25318495, 1.80652855, 1.]
        assert all(np.isclose(expected_p, self.p_values))

    def test_alpha_value(self):
        pass


class TestSAMcorrection:

    def test_single_value_input(self):
        f_values = 10
        curve_fold_change = 1
        s0 = 0
        f_adj = thresholding.sam_correction(f_values, curve_fold_change, s0)
        assert np.isclose(f_values, f_adj)

    def test_array_input(self):
        f_values = [10, 1, 0]
        curve_fold_changes = [1, 2, 3]
        s0s = [0, 0, 0]
        f_adjs = thresholding.sam_correction(f_values, curve_fold_changes, s0s)
        assert all(np.isclose(f_values, f_adjs))

    def test_different_s0(self):
        f_value = 100
        curve_fold_change = 1
        s0s = [np.inf, 0.9, 0.1, 0]
        expected_f_adjs = [  0,   1,  25, 100]
        f_adjs = thresholding.sam_correction(f_value, curve_fold_change, s0s)
        assert all(np.isclose(expected_f_adjs, f_adjs))

    def test_different_fold_changes(self):
        f_value = 100
        curve_fold_changes = [np.inf, 10, 1/.9, 0]
        s0 = 1
        expected_f_adjs = [100,  25,   1,   0]
        f_adjs = thresholding.sam_correction(f_value, curve_fold_changes, s0)
        assert all(np.isclose(expected_f_adjs, f_adjs))

    def test_different_fvalues(self):
        f_values = [np.inf, 400, 100, 25, 4, 0]
        curve_fold_change = 1
        s0 = 0.1
        expected_f_adjs = np.array([1/0.1,  1/0.15, 1/0.2, 1/0.3, 1/0.6, 0])**2
        f_adjs = thresholding.sam_correction(f_values, curve_fold_change, s0)
        assert all(np.isclose(expected_f_adjs, f_adjs))


class TestCalculateQValues:

    decoy_col = 'Decoy'
    sort_cols = ['Score']
    sort_ascendings = [False]
    q_col_name = 'q-value'

    def test_qvalues_all_one(self):
        # Create data
        target_df = get_empty_df(100, decoy=False)
        target_df['Score'] = np.linspace(0, 10, 100)
        decoy_df = get_empty_df(100, decoy=True)
        decoy_df['Score'] = np.linspace(0, 10, 100)
        df_in = pd.concat([target_df, decoy_df], ignore_index=True)
        # Apply function
        df_out = thresholding.calculate_qvalue(df_in, self.sort_cols, self.sort_ascendings, self.decoy_col, self.q_col_name)
        # Tests:
        assert self.q_col_name in df_out.columns
        expected_q = np.full(200, 1.0)
        assert all(np.isclose(expected_q, df_out[self.q_col_name].values))

    def test_qvalues_monotonic(self):
        # Create data
        target_df = get_empty_df(99, decoy=False)
        target_df['Score'] = np.array([np.linspace(1, 9, 9) for i in range(11)]).flatten()
        decoy_df = get_empty_df(99, decoy=True)
        decoy_df['Score'] = np.linspace(0.01, 0.99, 99)
        df_in = pd.concat([target_df, decoy_df], ignore_index=True)
        # Apply function
        df_out = thresholding.calculate_qvalue(df_in, self.sort_cols, self.sort_ascendings, self.decoy_col, self.q_col_name)
        # Tests
        expected_q = np.append(np.full(99, 0.01), np.linspace(1.0, 0.02, 99))
        assert all(np.isclose(expected_q, df_out[self.q_col_name].values))


class TestGetFDR:

    col = 'Score'
    threshold = 2
    target_decoy_ratio = 1.0

    def test_all_zero_score(self):
        target_df = get_empty_df(100, decoy=False)
        decoy_df = get_empty_df(100, decoy=True)

        expected_fdr = 1.0
        fdr = thresholding.get_fdr(target_df, decoy_df, self.col, self.threshold, self.target_decoy_ratio)
        assert np.isclose(expected_fdr, fdr)

    def test_all_pass_score(self):
        target_df = get_empty_df(100, decoy=False)
        target_df[self.col] = np.full(100, 3.0)
        decoy_df = get_empty_df(100, decoy=True)
        decoy_df[self.col] = np.full(100, 3.0)

        expected_fdr = 1.0
        fdr = thresholding.get_fdr(target_df, decoy_df, self.col, self.threshold, self.target_decoy_ratio)
        assert np.isclose(expected_fdr, fdr)

    def test_halve_pass_score(self):
        target_df = get_empty_df(100, decoy=False)
        target_df[self.col] = np.append(np.full(50, 3.0), np.full(50, 1.0))
        decoy_df = get_empty_df(100, decoy=True)
        decoy_df[self.col] = np.append(np.full(50, 3.0), np.full(50, 1.0))

        expected_fdr = 1.0
        fdr = thresholding.get_fdr(target_df, decoy_df, self.col, self.threshold, self.target_decoy_ratio)
        assert np.isclose(expected_fdr, fdr)

    def test_target_only(self):
        target_df = get_empty_df(100, decoy=False)
        target_df[self.col] = np.full(100, 3.0)
        decoy_df = get_empty_df(100, decoy=True)

        expected_fdr = 1/101
        fdr = thresholding.get_fdr(target_df, decoy_df, self.col, self.threshold, self.target_decoy_ratio)
        assert np.isclose(expected_fdr, fdr)

    def test_decoy_only(self):
        target_df = get_empty_df(100, decoy=False)
        decoy_df = get_empty_df(100, decoy=True)
        decoy_df[self.col] = np.full(100, 3.0)

        expected_fdr = 1.0
        fdr = thresholding.get_fdr(target_df, decoy_df, self.col, self.threshold, self.target_decoy_ratio)
        assert np.isclose(expected_fdr, fdr)

    def test_10pct_fdr(self):
        target_df = get_empty_df(99, decoy=False)
        target_df[self.col] = np.full(99, 3.0)
        decoy_df = get_empty_df(99, decoy=True)
        decoy_df[self.col] = np.append(np.full(9, 3.0), np.full(90, 1.0))

        expected_fdr = 0.1
        fdr = thresholding.get_fdr(target_df, decoy_df, self.col, self.threshold, self.target_decoy_ratio)
        assert np.isclose(expected_fdr, fdr)

    def test_taregt_decoy_ratio(self):
        target_df = get_empty_df(99, decoy=False)
        target_df[self.col] = np.full(99, 3.0)
        decoy_df = get_empty_df(299, decoy=True)
        decoy_df[self.col] = np.append(np.full(29, 3.0), np.full(270, 1.0))

        target_decoy_ratio = 1/3
        expected_fdr = 0.1
        fdr = thresholding.get_fdr(target_df, decoy_df, self.col, self.threshold, target_decoy_ratio)
        assert np.isclose(expected_fdr, fdr)



class TestDefineRegulatedCurves:
    pass
